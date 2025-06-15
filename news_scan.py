# news_scan.py

import os
import re
import requests
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
from dotenv import load_dotenv
import queue
from enum import Enum
import warnings
from truthbrush.api import Api as TruthbrushApi
warnings.filterwarnings('ignore')

# Audio and terminal enhancements
try:
    import pygame
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️  pygame not available. Audio alerts disabled. Install with: pip install pygame")

try:
    import colorama
    from colorama import Fore, Back, Style
    colorama.init()
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    print("⚠️  colorama not available. Colored output disabled. Install with: pip install colorama")

# Advanced NLP
try:
    from transformers.pipelines import pipeline
    ADVANCED_NLP_AVAILABLE = True
    print("✅ Transformers library loaded successfully")
except ImportError:
    ADVANCED_NLP_AVAILABLE = False
    print("⚠️  transformers not available. Using basic sentiment analysis only.")

# --- CONFIGURATION ---
def require_env(name: str) -> str:
    val = os.getenv(name)
    if val is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val

class Config:
    load_dotenv()
    NEWS_API_KEY      = require_env("NEWS_API_KEY")
    ALPHA_VANTAGE_KEY = require_env("ALPHA_VANTAGE_KEY")
    FINNHUB_KEY       = require_env("FINNHUB_KEY")
    
    # Analysis Parameters
    UPDATE_INTERVAL = 60  # seconds
    HISTORY_LENGTH = 100  # number of articles to keep in history
    SIGNAL_THRESHOLD = 0.3  # Lowered from 0.6 for more signals
    MAJOR_EVENT_THRESHOLD = 0.7  # Lowered from 0.8 for audio alerts
    
    # Asset Configuration
    ASSETS = {
        'GOLD': {
            'symbol': 'XAUUSD',
            'keywords': ['gold', 'xauusd', 'precious metal', 'inflation hedge', 'safe haven', 'bullion'],
            'weight': 1.0
        },
        'BITCOIN': {
            'symbol': 'BTCUSD',
            'keywords': ['bitcoin', 'btc', 'crypto', 'cryptocurrency', 'digital asset', 'blockchain'],
            'weight': 1.0
        },
        'SP500': {
            'symbol': 'SPX',
            'keywords': ['s&p 500', 'spx', 'stock market', 'equities', 'stocks'],
            'weight': 0.8
        },
        'USD': {
            'symbol': 'DXY',
            'keywords': ['dollar', 'usd', 'dxy', 'currency', 'federal reserve', 'fed'],
            'weight': 0.9
        }
    }

class SignalStrength(Enum):
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"
    CRITICAL = "CRITICAL"

class Direction(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

@dataclass
class Signal:
    asset: str
    direction: Direction
    strength: SignalStrength
    confidence: float
    reasoning: List[str]
    timestamp: datetime
    sources: List[str]
    sentiment_score: float
    market_impact: float

@dataclass
class MarketEvent:
    event_type: str
    severity: str
    description: str
    affected_assets: List[str]
    timestamp: datetime
    confidence: float

class AdvancedSentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.finbert = None
        self.model_name = None
        
        if ADVANCED_NLP_AVAILABLE:
            # List of models to try in order of preference
            models_to_try = [
                ("ProsusAI/finbert", "FinBERT Financial Model"),
                ("cardiffnlp/twitter-roberta-base-sentiment-latest", "RoBERTa Sentiment Model"),
                ("distilbert-base-uncased-finetuned-sst-2-english", "DistilBERT SST-2"),
                ("nlptown/bert-base-multilingual-uncased-sentiment", "BERT Multilingual Sentiment")
            ]
            
            for model_name, description in models_to_try:
                try:
                    print(f"🔄 Trying to load {description}...")
                    self.finbert = pipeline(
                        "sentiment-analysis", 
                        model=model_name,
                        device=-1,  # Force CPU
                        return_all_scores=False
                    )
                    self.model_name = model_name
                    print(f"✅ Successfully loaded {description}!")
                    break
                except Exception as e:
                    print(f"❌ Failed to load {model_name}: {str(e)[:100]}...")
                    continue
            
            if not self.finbert:
                print("⚠️  All transformer models failed to load. Using basic sentiment only.")
        else:
            print("⚠️  Transformers library not available")
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Comprehensive sentiment analysis using multiple methods"""
        # VADER analysis
        vader_scores = self.vader.polarity_scores(text)
        
        # TextBlob analysis
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # Advanced model analysis (if available)
        finbert_score = 0
        finbert_label = "neutral"
        
        if self.finbert:
            try:
                # Truncate text to avoid token limits
                truncated_text = text[:500] if len(text) > 500 else text
                result = self.finbert(truncated_text)
                
                # Handle different model output formats
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]
                
                finbert_label = result['label'].lower()
                finbert_score = result['score']
                
                # Normalize scores based on model type
                if self.model_name == "ProsusAI/finbert":
                    # FinBERT uses positive/negative/neutral
                    if finbert_label in ['negative', 'neg']:
                        finbert_score = -finbert_score
                    elif finbert_label in ['neutral']:
                        finbert_score = 0
                else:
                    # Other models might use different labels
                    if finbert_label in ['negative', 'neg', 'label_0']:
                        finbert_score = -finbert_score
                    elif finbert_label in ['neutral', 'label_1']:
                        finbert_score = 0
                    # positive stays positive
                
            except Exception as e:
                print(f"⚠️  Advanced model analysis failed: {e}")
                finbert_score = 0
                finbert_label = "error"
        
        # Weighted composite score
        if self.finbert:
            composite_score = (
                vader_scores['compound'] * 0.4 +
                textblob_polarity * 0.3 +
                finbert_score * 0.3
            )
        else:
            composite_score = (
                vader_scores['compound'] * 0.6 +
                textblob_polarity * 0.4
            )
        
        return {
            'composite': composite_score,
            'vader': vader_scores,
            'textblob': {'polarity': textblob_polarity, 'subjectivity': textblob_subjectivity},
            'finbert': {'score': finbert_score, 'label': finbert_label},
            'confidence': abs(composite_score),
            'model_used': self.model_name or 'VADER+TextBlob'
        }

class NewsDataProvider:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AdvancedFinancialAnalyzer/1.0'
        })
    
    def fetch_news_api(self) -> List[Dict]:
        """Fetch from NewsAPI"""
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': 'finance OR economy OR "federal reserve" OR inflation OR market',
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 50,
            'apiKey': Config.NEWS_API_KEY
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json().get('articles', [])
        except Exception as e:
            logging.error(f"NewsAPI error: {e}")
            return []
    
    def fetch_finnhub_news(self) -> List[Dict]:
        """Fetch from Finnhub"""
        url = "https://finnhub.io/api/v1/news"
        params = {
            'category': 'general',
            'token': Config.FINNHUB_KEY
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            articles = response.json()
            
            # Convert to standard format
            standardized = []
            for article in articles:
                standardized.append({
                    'title': article.get('headline', ''),
                    'description': article.get('summary', ''),
                    'url': article.get('url', ''),
                    'publishedAt': datetime.fromtimestamp(article.get('datetime', 0)).isoformat(),
                    'source': {'name': 'Finnhub'}
                })
            return standardized
        except Exception as e:
            logging.error(f"Finnhub error: {e}")
            return []
    
    def fetch_all_news(self) -> List[Dict]:
        """Fetch from all available sources"""
        all_articles = []
        
        if Config.NEWS_API_KEY and Config.NEWS_API_KEY != "YOUR_NEWS_API_KEY":
            all_articles.extend(self.fetch_news_api())
        
        if Config.FINNHUB_KEY and Config.FINNHUB_KEY != "YOUR_FINNHUB_KEY":
            all_articles.extend(self.fetch_finnhub_news())
        
        # Remove duplicates and sort by recency
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            title = article.get('title', '').lower()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append(article)
        
        return sorted(unique_articles, 
                     key=lambda x: x.get('publishedAt', ''), 
                     reverse=True)
    
class TruthSocialProvider:
    """
    A dedicated provider to fetch posts from Truth Social using its official API class.
    """
    def __init__(self):
        self.api_client = None  # We will store the Api object here
        username = os.getenv("TRUTHSOCIAL_USERNAME")
        password = os.getenv("TRUTHSOCIAL_PASSWORD")

        if not (username and password):
            print("⚠️  Truth Social credentials not found in .env file. Skipping this source.")
        else:
            try:
                print("🔄 Initializing Truth Social API client...")
                # Create an instance of the Api class from the library
                self.api_client = TruthbrushApi(username=username, password=password)
                # The login check happens automatically when we call a method,
                # so we don't need to do anything else here.
                print("✅ Truth Social provider is enabled.")
            except Exception as e:
                print(f"⚠️  Failed to initialize Truth Social client: {e}")
                self.api_client = None

    def fetch_user_posts(self, username: str, limit: int = 5) -> List[Dict]:
        """
        Fetches the latest posts from a specific user and standardizes them.
        """
        if not self.api_client:
            return []

        print(f"🔄 Fetching latest posts for '{username}' from Truth Social...")
        try:
            # Call the pull_statuses method on our api_client instance
            posts_generator = self.api_client.pull_statuses(username)
            
            collected_posts = []
            for i, post in enumerate(posts_generator):
                if i >= limit:
                    break
                collected_posts.append(post)

            if not collected_posts:
                print(f"   -> No new posts found for '{username}'.")
                return []
            
            print(f"   -> Found {len(collected_posts)} posts.")
            return self._standardize_posts(collected_posts)

        except Exception as e:
            print(f"⚠️  Failed to fetch Truth Social posts: {e}")
            # We can disable the client if a fetch fails to prevent repeated errors
            self.api_client = None
            return []

    def _standardize_posts(self, posts: List[Dict]) -> List[Dict]:
        """
        Converts Truth Social post format to our standard article format.
        """
        standardized_articles = []
        for post in posts:
            content = post.get('content', '')
            clean_content = re.sub('<[^<]+?>', ' ', content).strip()
            if not clean_content:
                continue

            article = {
                'title': clean_content[:70] + ("..." if len(clean_content) > 70 else ""),
                'description': clean_content,
                'url': post.get('url', ''),
                'publishedAt': post.get('created_at', ''),
                'source': {
                    'name': 'Truth Social',
                    'user': post.get('account', {}).get('username')
                }
            }
            standardized_articles.append(article)
        return standardized_articles

class AdvancedSignalGenerator:
    def __init__(self):
        self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        self.signal_history = defaultdict(deque)
        self.keyword_weights = self._build_keyword_weights()
        
    def _build_keyword_weights(self) -> Dict:
        """Build weighted keyword dictionary for market impact assessment"""
        return {
            # High impact economic indicators
            'inflation': 0.9, 'cpi': 0.9, 'ppi': 0.8, 'pce': 0.8,
            'unemployment': 0.8, 'nfp': 0.9, 'jobs': 0.7,
            'gdp': 0.9, 'recession': 0.9, 'growth': 0.7,
            
            # Central bank related
            'federal reserve': 0.9, 'fed': 0.8, 'powell': 0.8,
            'interest rate': 0.9, 'rate hike': 0.9, 'rate cut': 0.9,
            'monetary policy': 0.8, 'hawkish': 0.8, 'dovish': 0.8,
            
            # Geopolitical
            'war': 0.8, 'conflict': 0.7, 'sanctions': 0.7,
            'trade war': 0.8, 'tariff': 0.7,
            
            # Market specific
            'crash': 0.9, 'rally': 0.7, 'bubble': 0.8,
            'volatility': 0.6, 'correction': 0.7,
            
            # Crypto specific
            'regulation': 0.8, 'ban': 0.9, 'adoption': 0.7,
            'institutional': 0.7, 'etf': 0.8
        }
    
    def calculate_market_impact(self, text: str) -> float:
        """Calculate potential market impact based on keyword presence and weights"""
        text_lower = text.lower()
        total_impact = 0
        keyword_count = 0
        
        for keyword, weight in self.keyword_weights.items():
            if keyword in text_lower:
                total_impact += weight
                keyword_count += 1
        
        # Normalize impact score
        if keyword_count > 0:
            return min(total_impact / keyword_count, 1.0)
        return 0.1  # Base impact for financial news
    
    def _analyze_trump_post(self, content: str, sentiment: Dict, market_impact: float, article: Dict) -> List[Signal]:
        """
        A specialized analyzer for posts from Donald Trump's account.
        This uses a different, more direct set of rules.
        """
        signals = []
        content_lower = content.lower()

        # --- THEME 1: TRADE & TARIFFS (China & EU) ---
        # Keywords that signal new tariffs or escalating trade wars.
        trade_war_keywords = ['tariff', 'trade war', 'unfair trade', 'intellectual property theft', 'decouple', 'america first', 'china', 'european union', 'eu', 'china tariffs', 'trade deal', 'trade agreement', 'trade imbalance', 'trade deficit', 'protectionism', 'import duties']
        if any(word in content_lower for word in trade_war_keywords):
            # IMPACT: Bearish for SP500 (uncertainty), Bullish for Gold (safe haven), Neutral/Bearish for USD.
            signals.append(Signal(
                asset='SP500', direction=Direction.BEARISH, strength=self._determine_strength(0.85, market_impact),
                confidence=0.85, reasoning=["Potential for new tariffs or trade war escalation"],
                timestamp=datetime.now(), sources=[article.get('url', '')], sentiment_score=sentiment['composite'], market_impact=market_impact
            ))
            signals.append(Signal(
                asset='GOLD', direction=Direction.BULLISH, strength=self._determine_strength(0.80, market_impact),
                confidence=0.80, reasoning=["Trade uncertainty increases safe-haven demand"],
                timestamp=datetime.now(), sources=[article.get('url', '')], sentiment_score=sentiment['composite'], market_impact=market_impact
            ))

        # --- THEME 2: FEDERAL RESERVE & INTEREST RATES ---
        # Keywords related to criticizing or praising the Fed's policy.
        fed_criticism_keywords = ['fed', 'powell', 'interest rates', 'rate hike', 'too high', 'strong dollar']
        if any(word in content_lower for word in fed_criticism_keywords):
            # IMPACT: Usually implies he wants lower rates, which is Bullish for SP500 and Gold.
            signals.append(Signal(
                asset='SP500', direction=Direction.BULLISH, strength=self._determine_strength(0.75, market_impact),
                confidence=0.75, reasoning=["Hints at desire for lower interest rates"],
                timestamp=datetime.now(), sources=[article.get('url', '')], sentiment_score=sentiment['composite'], market_impact=market_impact
            ))
            signals.append(Signal(
                asset='GOLD', direction=Direction.BULLISH, strength=self._determine_strength(0.80, market_impact),
                confidence=0.80, reasoning=["Lower rates increase gold's appeal as hedge"],
                timestamp=datetime.now(), sources=[article.get('url', '')], sentiment_score=sentiment['composite'], market_impact=market_impact
            ))
            signals.append(Signal(
                asset='USD', direction=Direction.BEARISH, strength=self._determine_strength(0.70, market_impact),
                confidence=0.70, reasoning=["Pressure on Fed could weaken the dollar"],
                timestamp=datetime.now(), sources=[article.get('url', '')], sentiment_score=sentiment['composite'], market_impact=market_impact
            ))

        # --- THEME 3: DOMESTIC ECONOMY (Praise or Alarm) ---
        # Keywords boasting about economic performance.
        economy_praise_keywords = ['strongest economy', 'best ever', 'roaring back', 'tax cuts', 'low taxes', 'lower tax', 'jobs jobs jobs', 'deregulation']
        if any(word in content_lower for word in economy_praise_keywords):
            # IMPACT: Directly Bullish for the stock market (SP500).
            signals.append(Signal(
                asset='SP500', direction=Direction.BULLISH, strength=self._determine_strength(0.80, market_impact),
                confidence=0.90, reasoning=["Positive commentary on US economic strength"],
                timestamp=datetime.now(), sources=[article.get('url', '')], sentiment_score=sentiment['composite'], market_impact=market_impact
            ))

        # --- THEME 4: CRYPTOCURRENCY & BITCOIN ---
        crypto_keywords = ['crypto', 'bitcoin', 'btc', 'cryptocurrency', 'digital asset', 'blockchain', 'decentralized finance', 'ethereum', 'web3', 'nft', 'meme coin']
        bullish_keywords = ['bullish', 'rally', 'uptrend', 'soaring', 'optimistic', 'buy', 'adoption', 'growth']
        bearish_keywords = ['bearish', 'crash', 'drop', 'decline', 'pessimistic', 'sell', 'downtrend']

        if any(word in content_lower for word in crypto_keywords):
            # Check for bullish sentiment keywords first
            if any(word in content_lower for word in bullish_keywords):
                signals.append(Signal(
                    asset='BITCOIN', direction=Direction.BULLISH, strength=self._determine_strength(0.80, market_impact),
                    confidence=0.80, reasoning=["Positive sentiment on cryptocurrencies"],
                    timestamp=datetime.now(), sources=[article.get('url', '')], sentiment_score=sentiment['composite'], market_impact=market_impact
                ))
            # Check for bearish sentiment keywords
            elif any(word in content_lower for word in bearish_keywords):
                signals.append(Signal(
                    asset='BITCOIN', direction=Direction.BEARISH, strength=self._determine_strength(0.80, market_impact),
                    confidence=0.80, reasoning=["Negative sentiment on cryptocurrencies"],
                    timestamp=datetime.now(), sources=[article.get('url', '')], sentiment_score=sentiment['composite'], market_impact=market_impact
                ))
            # Fallback: if no specific keyword match, decide based on overall sentiment
            else:
                direction = Direction.BULLISH if sentiment['composite'] >= 0 else Direction.BEARISH
                reasoning_text = "Overall positive sentiment on cryptocurrencies" if direction == Direction.BULLISH else "Overall negative sentiment on cryptocurrencies"
                signals.append(Signal(
                    asset='BITCOIN', direction=direction, strength=self._determine_strength(0.80, market_impact),
                    confidence=0.80, reasoning=[reasoning_text],
                    timestamp=datetime.now(), sources=[article.get('url', '')], sentiment_score=sentiment['composite'], market_impact=market_impact
                ))
            
        # --- THEME 5: GEOPOLITICAL INSTABILITY ---
        # Keywords related to global conflicts that are not direct trade wars.
        geopolitical_keywords = ['nato', 'ukraine', 'russia', 'middle east', 'iran', 'china', 'taiwan', 'israel', 'north korea', 'conflict', 'war', 'crisis']
        if any(word in content_lower for word in geopolitical_keywords):
            # IMPACT: General uncertainty is a flight to safety, bullish for USD and Gold.
            signals.append(Signal(
                asset='GOLD', direction=Direction.BULLISH, strength=self._determine_strength(0.80, market_impact),
                confidence=0.80, reasoning=["Geopolitical uncertainty drives safe-haven demand"],
                timestamp=datetime.now(), sources=[article.get('url', '')], sentiment_score=sentiment['composite'], market_impact=market_impact
            ))
            signals.append(Signal(
                asset='USD', direction=Direction.BULLISH, strength=self._determine_strength(0.70, market_impact),
                confidence=0.70, reasoning=["USD seen as safe haven during global instability"],
                timestamp=datetime.now(), sources=[article.get('url', '')], sentiment_score=sentiment['composite'], market_impact=market_impact
            ))
            signals.append(Signal(
                asset='SP500', direction=Direction.BEARISH, strength=self._determine_strength(0.75, market_impact),
                confidence=0.75, reasoning=["Geopolitical tensions create market uncertainty"],
                timestamp=datetime.now(), sources=[article.get('url', '')], sentiment_score=sentiment['composite'], market_impact=market_impact
            ))

        return signals
    
    def generate_signals(self, article: Dict) -> List[Signal]:
        """Generate trading signals from news article"""
        title = article.get('title', '')
        description = article.get('description', '')
        content = f"{title}. {description}"
        source = article.get('source', {})
        source_name = source.get('name')
        source_user = source.get('user')

        # Sentiment analysis
        sentiment_data = self.sentiment_analyzer.analyze_sentiment(content)

        # Check if the post is from Trump on Truth Social
        if source_name == 'Truth Social' and source_user == 'realDonaldTrump':
            print("✅ Detected post from Trump. Applying specialized analysis.")
            # Manually set a very high impact score
            market_impact = 0.90
            # Route to the specialist function and return its signals directly
            return self._analyze_trump_post(content, sentiment_data, market_impact, article)

        # If it's not a Trump post, proceed with the general analysis
        market_impact = self.calculate_market_impact(content)
        signals = []
        for asset_name, asset_config in Config.ASSETS.items():
            signal = self._analyze_asset_impact(
                asset_name, asset_config, content,
                sentiment_data, market_impact, article
            )
            if signal:
                signals.append(signal)

        return signals
    
    def _analyze_asset_impact(self, asset_name: str, asset_config: Dict, 
                            content: str, sentiment_data: Dict, 
                            market_impact: float, article: Dict) -> Optional[Signal]:
        """Analyze impact on specific asset"""
        content_lower = content.lower()
        
        # Check keyword relevance
        keyword_matches = sum(1 for keyword in asset_config['keywords'] 
                            if keyword in content_lower)
        
        if keyword_matches == 0 and market_impact < 0.5:
            return None  # Not relevant to this asset
        
        # Asset-specific analysis
        direction = Direction.NEUTRAL
        reasoning = []
        confidence = sentiment_data['confidence'] * asset_config['weight']
        
        if asset_name == 'GOLD':
            direction, reasoning = self._analyze_gold(content_lower, sentiment_data)
        elif asset_name == 'BITCOIN':
            direction, reasoning = self._analyze_bitcoin(content_lower, sentiment_data)
        elif asset_name == 'SP500':
            direction, reasoning = self._analyze_sp500(content_lower, sentiment_data)
        elif asset_name == 'USD':
            direction, reasoning = self._analyze_usd(content_lower, sentiment_data)
        
        # Determine signal strength
        strength = self._determine_strength(confidence, market_impact)
        
        final_sentiment_score = sentiment_data['composite']
        
        # If  rules determined a bullish direction despite negative news sentiment,
        # reflect that in the score displayed.
        if direction == Direction.BULLISH and final_sentiment_score < 0:
            final_sentiment_score = abs(final_sentiment_score)
        
        # If rules determined a bearish direction despite positive news sentiment,
        # reflect that in the score.
        elif direction == Direction.BEARISH and final_sentiment_score > 0:
            final_sentiment_score = -abs(final_sentiment_score)

        if direction == Direction.NEUTRAL and strength == SignalStrength.WEAK:
            return None
        
        return Signal(
            asset=asset_name,
            direction=direction,
            strength=strength,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=datetime.now(),
            sources=[article.get('url', '')],
            sentiment_score=final_sentiment_score, # <-- Use the adjusted score
            market_impact=market_impact
        )
    
    def _analyze_gold(self, content: str, sentiment: Dict) -> Tuple[Direction, List[str]]:
        """Gold-specific analysis logic"""
        reasoning = []
        direction = Direction.NEUTRAL
        
        # Inflation hedge logic
        if any(word in content for word in ['inflation', 'cpi', 'ppi']):
            if 'high' in content or 'rising' in content or sentiment['composite'] > 0.3:
                direction = Direction.BULLISH
                reasoning.append("Rising inflation increases gold's appeal as hedge")
        
        # Safe haven demand
        if any(word in content for word in ['uncertainty', 'crisis', 'war', 'conflict']):
            direction = Direction.BULLISH
            reasoning.append("Geopolitical uncertainty drives safe-haven demand")
        
        # Interest rate impact
        if any(word in content for word in ['rate hike', 'hawkish', 'tightening']):
            direction = Direction.BEARISH
            reasoning.append("Higher rates increase opportunity cost of holding gold")
        
        # Dollar strength
        if 'dollar strength' in content or 'strong dollar' in content:
            direction = Direction.BEARISH
            reasoning.append("Strong dollar makes gold more expensive for foreign buyers")
        
        return direction, reasoning
    
    def _analyze_bitcoin(self, content: str, sentiment: Dict) -> Tuple[Direction, List[str]]:
        """Bitcoin-specific analysis logic"""
        reasoning = []
        direction = Direction.NEUTRAL
        
        # Regulatory news
        if any(word in content for word in ['regulation', 'ban', 'legal']):
            if sentiment['composite'] < -0.2:
                direction = Direction.BEARISH
                reasoning.append("Negative regulatory developments")
            elif sentiment['composite'] > 0.2:
                direction = Direction.BULLISH
                reasoning.append("Positive regulatory clarity")
        
        # Institutional adoption
        if any(word in content for word in ['institutional', 'etf', 'adoption', 'corporate']):
            if sentiment['composite'] > 0.1:
                direction = Direction.BULLISH
                reasoning.append("Institutional adoption increases legitimacy")
        
        # Risk-off sentiment
        if any(word in content for word in ['risk-off', 'uncertainty', 'volatility']):
            direction = Direction.BEARISH
            reasoning.append("Risk-off sentiment affects crypto markets")
        
        # Technology developments
        if any(word in content for word in ['blockchain', 'innovation', 'upgrade']):
            if sentiment['composite'] > 0.2:
                direction = Direction.BULLISH
                reasoning.append("Positive technology developments")
        
        return direction, reasoning
    
    def _analyze_sp500(self, content: str, sentiment: Dict) -> Tuple[Direction, List[str]]:
        """S&P 500 specific analysis logic"""
        reasoning = []
        direction = Direction.NEUTRAL
        
        # Economic growth
        if any(word in content for word in ['gdp', 'growth', 'expansion']):
            if sentiment['composite'] > 0.2:
                direction = Direction.BULLISH
                reasoning.append("Strong economic growth supports equities")
            elif sentiment['composite'] < -0.2:
                direction = Direction.BEARISH
                reasoning.append("Economic slowdown concerns")
        
        # Fed policy
        if any(word in content for word in ['fed', 'federal reserve', 'powell']):
            if 'dovish' in content or 'rate cut' in content:
                direction = Direction.BULLISH
                reasoning.append("Accommodative Fed policy supports risk assets")
            elif 'hawkish' in content or 'rate hike' in content:
                direction = Direction.BEARISH
                reasoning.append("Tighter monetary policy pressures valuations")
        
        # Earnings and corporate news
        if any(word in content for word in ['earnings', 'profit', 'revenue']):
            if sentiment['composite'] > 0.3:
                direction = Direction.BULLISH
                reasoning.append("Strong corporate fundamentals")
        
        return direction, reasoning
    
    def _analyze_usd(self, content: str, sentiment: Dict) -> Tuple[Direction, List[str]]:
        """USD specific analysis logic"""
        reasoning = []
        direction = Direction.NEUTRAL
        
        # Fed policy impact
        if any(word in content for word in ['fed', 'federal reserve']):
            if 'hawkish' in content or 'rate hike' in content:
                direction = Direction.BULLISH
                reasoning.append("Hawkish Fed policy strengthens dollar")
            elif 'dovish' in content or 'rate cut' in content:
                direction = Direction.BEARISH
                reasoning.append("Dovish Fed policy weakens dollar")
        
        # Economic data
        if any(word in content for word in ['nfp', 'unemployment', 'jobs']):
            if sentiment['composite'] > 0.2:
                direction = Direction.BULLISH
                reasoning.append("Strong employment data supports USD")
        
        return direction, reasoning
    
    def _determine_strength(self, confidence: float, market_impact: float) -> SignalStrength:
        """Determine signal strength based on confidence and market impact"""
        combined_score = (confidence + market_impact) / 2
        
        if combined_score >= 0.8:
            return SignalStrength.CRITICAL
        elif combined_score >= 0.6:
            return SignalStrength.STRONG
        elif combined_score >= 0.3:  # Lowered from 0.4
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK

class AudioAlertManager:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_initialized = False
        
        if AUDIO_AVAILABLE:
            try:
                pygame.mixer.init()
                self.is_initialized = True
                # Generate alert tones
                self._generate_alert_sounds()
            except Exception as e:
                print(f"⚠️  Audio initialization failed: {e}")
    
    def _generate_alert_sounds(self):
        """Generate different alert tones for different signal strengths"""
        # This would generate different frequency tones
        # For simplicity, we'll use system beep or pre-recorded sounds
        pass
    
    def play_alert(self, signal_strength: SignalStrength):
        """Play audio alert based on signal strength"""
        if not self.is_initialized:
            return
        
        # Simple system beep for now
        if signal_strength in [SignalStrength.STRONG, SignalStrength.CRITICAL]:
            print('\a')  # System beep

class TerminalDisplay:
    def __init__(self):
        self.use_colors = COLORS_AVAILABLE
    
    def print_header(self):
        """Print application header"""
        header = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ADVANCED FINANCIAL MARKET ANALYZER                        ║
║                          Real-time Signal Generation                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        if self.use_colors:
            print(Fore.CYAN + Style.BRIGHT + header + Style.RESET_ALL)
        else:
            print(header)
    
    def print_signal(self, signal: Signal):
        """Print signal with formatting"""
        timestamp = signal.timestamp.strftime("%H:%M:%S")
        
        # Color coding based on direction and strength
        if self.use_colors:
            if signal.direction == Direction.BULLISH:
                direction_color = Fore.GREEN
            elif signal.direction == Direction.BEARISH:
                direction_color = Fore.RED
            else:
                direction_color = Fore.YELLOW
            
            strength_color = Fore.WHITE
            if signal.strength == SignalStrength.CRITICAL:
                strength_color = Fore.MAGENTA + Style.BRIGHT
            elif signal.strength == SignalStrength.STRONG:
                strength_color = Fore.YELLOW + Style.BRIGHT
            
            print(f"\n[{Fore.CYAN}{timestamp}{Style.RESET_ALL}] "
                  f"{direction_color}{signal.direction.value}{Style.RESET_ALL} "
                  f"{strength_color}{signal.strength.value}{Style.RESET_ALL} "
                  f"signal for {Fore.WHITE + Style.BRIGHT}{signal.asset}{Style.RESET_ALL}")
        else:
            print(f"\n[{timestamp}] {signal.direction.value} {signal.strength.value} signal for {signal.asset}")
        
        print(f"  Confidence: {signal.confidence:.2f} | Market Impact: {signal.market_impact:.2f}")
        print(f"  Sentiment Score: {signal.sentiment_score:.2f}")
        
        for reason in signal.reasoning:
            print(f"  • {reason}")
        
        if signal.strength in [SignalStrength.STRONG, SignalStrength.CRITICAL]:
            border = "!" * 60
            if self.use_colors:
                print(Fore.RED + Style.BRIGHT + border + Style.RESET_ALL)
            else:
                print(border)
    
    def print_summary(self, analyzer_instance): # <-- Note the change in argument
        """
        Print a summary of the market state. This version looks at all signals
        generated in the last 15 minutes and prioritizes the most critical ones.
        """
        all_recent_signals = analyzer_instance.signal_history
        
        lookback_window = timedelta(minutes=15)
        now = datetime.now()
        signals_in_window = [
            s for s in all_recent_signals
            if (now - s.timestamp) <= lookback_window
        ]        

        # Header
        header_color = Fore.CYAN + Style.BRIGHT
        reset = Style.RESET_ALL
        print(f"\n{header_color}╔══════════════════════════════════════════════════════════════╗{reset}")
        print(f"{header_color}║           MARKET STATE SUMMARY (Last 15 Minutes)             ║{reset}")
        print(f"{header_color}║                  {now.strftime('%Y-%m-%d %H:%M:%S')}                     ║{reset}")
        print(f"{header_color}╠════════════╤══════════════════╤══════════════════╤════════════╣{reset}")
        print(f"{header_color}║  ASSET     │    DIRECTION     │     STRENGTH     │ CONFIDENCE ║{reset}")
        print(f"{header_color}╠════════════╪══════════════════╪══════════════════╪════════════╣{reset}")
        
        if not signals_in_window:
            print(f"{header_color}║          No significant signals in the last 15 minutes.        ║{reset}")
            print(f"{header_color}╚══════════════════════════════════════════════════════════════╝{reset}")
            return

        by_asset = defaultdict(list)
        for signal in signals_in_window:
            by_asset[signal.asset].append(signal)

        strength_map = {
            SignalStrength.WEAK: 1, SignalStrength.MODERATE: 2,
            SignalStrength.STRONG: 3, SignalStrength.CRITICAL: 4
        }

        # Body of the table
        for asset in Config.ASSETS.keys():
            if asset in by_asset:
                asset_signals = by_asset[asset]
                sig = max(asset_signals, key=lambda s: (strength_map[s.strength], s.timestamp))
                
                # Determine colors based on direction and strength
                dir_color = Fore.YELLOW  # Neutral
                if sig.direction == Direction.BULLISH: dir_color = Fore.GREEN + Style.BRIGHT
                elif sig.direction == Direction.BEARISH: dir_color = Fore.RED + Style.BRIGHT
                
                str_color = Fore.WHITE # Default/Weak
                if sig.strength == SignalStrength.MODERATE: str_color = Fore.CYAN
                elif sig.strength == SignalStrength.STRONG: str_color = Fore.YELLOW + Style.BRIGHT
                elif sig.strength == SignalStrength.CRITICAL: str_color = Fore.MAGENTA + Style.BRIGHT

                asset_str = f"{Fore.WHITE}{asset:^10}{reset}"
                dir_str = f"{dir_color}{sig.direction.value:^16}{reset}"
                str_str = f"{str_color}{sig.strength.value:^16}{reset}"
                conf_str = f"{Fore.WHITE}{sig.confidence:^10.2f}{reset}"

                print(f"{header_color}║ {asset_str} │ {dir_str} │ {str_str} │ {conf_str} ║{reset}")

            else:
                # Handle assets with no recent signals
                asset_str = f"{Fore.WHITE}{asset:^10}{reset}"
                neutral_str = f"{Fore.DIM}{'NEUTRAL':^16}{reset}"
                no_signal_str = f"{Fore.DIM}{'-':^16}{reset}"
                no_conf_str = f"{Fore.DIM}{'---':^10}{reset}"
                print(f"{header_color}║ {asset_str} │ {neutral_str} │ {no_signal_str} │ {no_conf_str} ║{reset}")
        
        # Footer
        print(f"{header_color}╚════════════╧══════════════════╧══════════════════╧════════════╝{reset}")

class AdvancedFinancialAnalyzer:
    def __init__(self):
        self.news_provider = NewsDataProvider()
        self.signal_generator = AdvancedSignalGenerator()
        self.truth_social_provider = TruthSocialProvider()
        self.audio_manager = AudioAlertManager()
        self.terminal = TerminalDisplay()
        
        # Data storage
        self.article_history = deque(maxlen=Config.HISTORY_LENGTH)
        self.signal_history = deque(maxlen=Config.HISTORY_LENGTH)
        self.processed_urls = set()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('market_analyzer.log'),
                logging.StreamHandler()
            ]
        )
        
        self.terminal.print_header()
    
    def run_terminal_mode(self):
        """Run in terminal mode with continuous updates"""
        print("🚀 Starting terminal mode... Press Ctrl+C to exit")
        
        try:
            while True:
                self._update_and_analyze()
                time.sleep(Config.UPDATE_INTERVAL)
        except KeyboardInterrupt:
            print("\n👋 Shutting down analyzer...")
            logging.info("Application terminated by user")
    
    def _update_and_analyze(self):
        """Fetch latest news and generate signals"""
        print(f"\n🔄 Fetching news... {datetime.now().strftime('%H:%M:%S')}")
        
        # Fetch news
        articles = self.news_provider.fetch_all_news()

        # Fetch posts from Truth Social and add them to the list
        if self.truth_social_provider.api_client:
            trump_posts = self.truth_social_provider.fetch_user_posts("realDonaldTrump", limit=5)
            articles.extend(trump_posts)

        new_articles = []
        
        for article in articles:
            url = article.get('url', '')
            if url and url not in self.processed_urls:
                new_articles.append(article)
                self.processed_urls.add(url)
                self.article_history.append(article)
        
        if not new_articles:
            print("📰 No new articles found")
            return
        
        print(f"📈 Analyzing {len(new_articles)} new articles...")
        
        # Generate signals
        all_signals = []
        for article in new_articles:
            signals = self.signal_generator.generate_signals(article)
            all_signals.extend(signals)
        
        # Filter significant signals
        significant_signals = [
            s for s in all_signals 
            if s.confidence >= Config.SIGNAL_THRESHOLD
        ]

        strength_map = {
            SignalStrength.WEAK: 1, SignalStrength.MODERATE: 2,
            SignalStrength.STRONG: 3, SignalStrength.CRITICAL: 4
        }
        
        # Sort signals by strength and confidence
        significant_signals.sort(key=lambda s: (strength_map[s.strength], s.confidence))
        
        if significant_signals:
            print(f"\n🎯 Generated {len(significant_signals)} signals:")
            
            for signal in significant_signals:
                self.terminal.print_signal(signal)
                self.signal_history.append(signal)
                
                # Audio alert for major events
                if signal.confidence >= Config.MAJOR_EVENT_THRESHOLD:
                    self.audio_manager.play_alert(signal.strength)
            
            self.terminal.print_summary(self)
        else:
            print("📊 No significant signals generated")
    
    def run_streamlit_dashboard(self):
        """Run Streamlit dashboard (enhanced version)"""
        st.set_page_config(
            page_title="Advanced Market Analyzer",
            page_icon="📈",
            layout="wide"
        )
        
        st.title("🚀 Advanced Financial Market Analyzer")
        st.markdown("Real-time market sentiment analysis and signal generation")
        
        # Sidebar controls
        with st.sidebar:
            st.header("⚙️ Configuration")
            update_interval = st.slider("Update Interval (seconds)", 30, 300, Config.UPDATE_INTERVAL)
            signal_threshold = st.slider("Signal Threshold", 0.1, 1.0, Config.SIGNAL_THRESHOLD)
            
            st.header("📊 Statistics")
            st.metric("Articles Processed", len(self.article_history))
            st.metric("Signals Generated", len(self.signal_history))
        
        # Main dashboard
        placeholder = st.empty()
        
        while True:
            with placeholder.container():
                # Fetch and analyze
                articles = self.news_provider.fetch_all_news()
                
                if articles:
                    # Display latest signals
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("🎯 Latest Signals")
                        
                        for article in articles[:5]:
                            signals = self.signal_generator.generate_signals(article)
                            significant = [s for s in signals if s.confidence >= signal_threshold]
                            
                            if significant:
                                st.markdown(f"**[{article['title']}]({article['url']})**")
                                
                                cols = st.columns(len(Config.ASSETS))
                                for i, (asset_name, _) in enumerate(Config.ASSETS.items()):
                                    asset_signals = [s for s in significant if s.asset == asset_name]
                                    
                                    with cols[i]:
                                        if asset_signals:
                                            signal = asset_signals[0]
                                            direction_emoji = "🟢" if signal.direction == Direction.BULLISH else "🔴" if signal.direction == Direction.BEARISH else "🟡"
                                            st.metric(
                                                asset_name,
                                                f"{direction_emoji} {signal.direction.value}",
                                                f"{signal.confidence:.2f}"
                                            )
                                        else:
                                            st.metric(asset_name, "➖ NEUTRAL", "0.00")
                                
                                st.markdown("---")
                    
                    with col2:
                        st.subheader("📈 Signal History")
                        if self.signal_history:
                            # Create DataFrame for visualization
                            signal_data = []
                            for signal in list(self.signal_history)[-20:]:
                                signal_data.append({
                                    'Time': signal.timestamp,
                                    'Asset': signal.asset,
                                    'Direction': signal.direction.value,
                                    'Confidence': signal.confidence
                                })
                            
                            df = pd.DataFrame(signal_data)
                            if not df.empty:
                                st.dataframe(df, use_container_width=True)
            
            time.sleep(update_interval)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Financial Market Analyzer")
    parser.add_argument('--mode', choices=['terminal', 'dashboard'], default='terminal',
                       help='Run mode: terminal or dashboard')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = AdvancedFinancialAnalyzer()
    
    if args.mode == 'terminal':
        analyzer.run_terminal_mode()
    else:
        analyzer.run_streamlit_dashboard()

if __name__ == "__main__":
    main()
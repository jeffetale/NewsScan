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
        """Comprehensive sentiment analysis with improved error handling and normalization"""
        # VADER analysis
        vader_scores = self.vader.polarity_scores(text)
        
        # TextBlob analysis
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # Advanced model analysis (if available)
        finbert_score = 0
        finbert_label = "neutral"
        finbert_confidence = 0
        
        if self.finbert:
            try:
                # Truncate and clean text for better processing
                truncated_text = text[:500] if len(text) > 500 else text
                # Remove excessive whitespace and newlines
                cleaned_text = ' '.join(truncated_text.split())
                
                result = self.finbert(cleaned_text)
                
                # Handle different model output formats
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]
                
                finbert_label = result['label'].lower()
                finbert_confidence = result.get('score', 0)
                
                # Enhanced normalization based on model type with confidence weighting
                if self.model_name == "ProsusAI/finbert":
                    # FinBERT uses positive/negative/neutral
                    if finbert_label in ['negative', 'neg']:
                        finbert_score = -finbert_confidence
                    elif finbert_label in ['positive', 'pos']:
                        finbert_score = finbert_confidence
                    else:  # neutral
                        finbert_score = 0
                else:
                    # Other models mapping
                    label_mapping = {
                        'negative': -finbert_confidence,
                        'neg': -finbert_confidence,
                        'label_0': -finbert_confidence,
                        'positive': finbert_confidence,
                        'pos': finbert_confidence,
                        'label_2': finbert_confidence,
                        'neutral': 0,
                        'label_1': 0
                    }
                    finbert_score = label_mapping.get(finbert_label, 0)
                
            except Exception as e:
                print(f"⚠️  Advanced model analysis failed: {e}")
                finbert_score = 0
                finbert_label = "error"
                finbert_confidence = 0
        
        # Enhanced weighted composite score with confidence adjustment
        if self.finbert and finbert_confidence > 0.5:  # Only use FinBERT if confident
            composite_score = (
                vader_scores['compound'] * 0.35 +
                textblob_polarity * 0.25 +
                finbert_score * 0.40
            )
            confidence_base = (abs(vader_scores['compound']) + abs(textblob_polarity) + finbert_confidence) / 3
        else:
            composite_score = (
                vader_scores['compound'] * 0.65 +
                textblob_polarity * 0.35
            )
            confidence_base = (abs(vader_scores['compound']) + abs(textblob_polarity)) / 2
        
        # Apply subjectivity adjustment - more subjective text gets lower confidence
        subjectivity_penalty = textblob_subjectivity * 0.2
        final_confidence = max(0.1, confidence_base - subjectivity_penalty)
        
        return {
            'composite': composite_score,
            'vader': vader_scores,
            'textblob': {'polarity': textblob_polarity, 'subjectivity': textblob_subjectivity},
            'finbert': {'score': finbert_score, 'label': finbert_label, 'confidence': finbert_confidence},
            'confidence': final_confidence,
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
        """Build comprehensive weighted keyword dictionary for market impact assessment"""
        return {
            # TIER 1: CRITICAL MARKET MOVERS (0.85-1.0)
            # Federal Reserve & Monetary Policy
            'fomc': 1.0, 'federal reserve': 0.95, 'fed meeting': 1.0, 'powell': 0.9,
            'interest rate': 0.95, 'rate hike': 0.95, 'rate cut': 0.95, 'fed funds': 0.9,
            'quantitative easing': 0.9, 'tapering': 0.9, 'balance sheet': 0.85,
            'dot plot': 0.9, 'jackson hole': 0.85, 'fed minutes': 0.85,
            
            # Critical Economic Indicators
            'nonfarm payrolls': 1.0, 'nfp': 1.0, 'unemployment rate': 0.9,
            'inflation': 0.95, 'cpi': 0.95, 'core cpi': 0.95, 'ppi': 0.85,
            'pce': 0.9, 'core pce': 0.9, 'gdp': 0.9, 'recession': 1.0,
            'jobless claims': 0.85, 'ism manufacturing': 0.85, 'ism services': 0.8,
            
            # Market Crisis Terms
            'market crash': 1.0, 'flash crash': 0.95, 'circuit breaker': 0.9,
            'vix spike': 0.85, 'volatility surge': 0.8, 'liquidity crisis': 0.95,
            'margin call': 0.85, 'forced selling': 0.8,
            
            # TIER 2: HIGH IMPACT (0.7-0.84)
            # Central Bank Communications
            'hawkish': 0.8, 'dovish': 0.8, 'monetary policy': 0.8,
            'yellen': 0.75, 'brainard': 0.7, 'williams': 0.7, 'clarida': 0.7,
            'kashkari': 0.7, 'bullard': 0.75, 'fed chair': 0.8,
            
            # Economic Data
            'retail sales': 0.8, 'consumer confidence': 0.75, 'consumer sentiment': 0.75,
            'housing starts': 0.75, 'existing home sales': 0.7, 'new home sales': 0.7,
            'durable goods': 0.75, 'factory orders': 0.7, 'industrial production': 0.75,
            'capacity utilization': 0.7, 'productivity': 0.7, 'unit labor costs': 0.7,
            
            # Geopolitical & Trade
            'trade war': 0.8, 'tariff': 0.8, 'sanctions': 0.75, 'trade deal': 0.8,
            'wto': 0.7, 'usmca': 0.7, 'brexit': 0.8, 'china trade': 0.8,
            'supply chain': 0.75, 'chip shortage': 0.75, 'energy crisis': 0.8,
            
            # Market Structure
            'earnings season': 0.8, 'guidance': 0.75, 'buyback': 0.7, 'dividend': 0.7,
            'ipo': 0.7, 'merger': 0.75, 'acquisition': 0.75, 'spinoff': 0.7,
            'activist investor': 0.7, 'short squeeze': 0.8, 'gamma squeeze': 0.75,
            
            # TIER 3: MODERATE IMPACT (0.5-0.69)
            # Regional Economics
            'eurozone': 0.65, 'ecb': 0.7, 'lagarde': 0.65, 'bank of england': 0.6,
            'boe': 0.6, 'bank of japan': 0.6, 'boj': 0.6, 'people\'s bank of china': 0.65,
            'pboc': 0.65, 'yield curve': 0.65, 'inverted yield curve': 0.8,
            
            # Sector Specific
            'tech earnings': 0.65, 'faang': 0.6, 'magnificent seven': 0.65,
            'semiconductor': 0.6, 'energy sector': 0.6, 'financials': 0.6,
            'healthcare': 0.55, 'utilities': 0.5, 'consumer staples': 0.5,
            
            # Crypto & Alternative Assets
            'bitcoin etf': 0.8, 'crypto regulation': 0.75, 'defi': 0.6,
            'stablecoin': 0.65, 'cbdc': 0.7, 'crypto ban': 0.8, 'mining ban': 0.7,
            'sec crypto': 0.75, 'gensler': 0.7, 'coinbase': 0.6,
            
            # Commodities
            'oil price': 0.7, 'crude oil': 0.7, 'opec': 0.75, 'natural gas': 0.65,
            'gold price': 0.6, 'silver': 0.5, 'copper': 0.6, 'wheat': 0.55,
            'corn': 0.5, 'agricultural': 0.55,
            
            # TIER 4: LOWER IMPACT (0.3-0.49)
            # General Market Terms
            'stock market': 0.4, 'equity': 0.4, 'bond market': 0.45, 'treasury': 0.45,
            'yield': 0.4, 'spread': 0.4, 'volatility': 0.45, 'volume': 0.35,
            'correlation': 0.3, 'beta': 0.3, 'alpha': 0.3,
            
            # General Economic Terms
            'economy': 0.4, 'economic': 0.4, 'growth': 0.45, 'expansion': 0.4,
            'contraction': 0.45, 'recovery': 0.4, 'slowdown': 0.45,
            'stimulus': 0.6, 'fiscal policy': 0.55, 'tax cut': 0.5, 'tax increase': 0.5,
            
            # Market Sentiment
            'bullish': 0.35, 'bearish': 0.35, 'rally': 0.4, 'sell-off': 0.45,
            'correction': 0.55, 'bear market': 0.7, 'bull market': 0.6,
            'support level': 0.3, 'resistance level': 0.3, 'breakout': 0.35,
            
            # Technical Analysis
            'moving average': 0.3, 'rsi': 0.25, 'macd': 0.25, 'bollinger bands': 0.25,
            'fibonacci': 0.25, 'trend line': 0.3, 'chart pattern': 0.3,
            
            # Company Specific (when they move markets)
            'apple': 0.4, 'microsoft': 0.4, 'amazon': 0.4, 'google': 0.4, 'tesla': 0.5,
            'nvidia': 0.5, 'meta': 0.45, 'netflix': 0.4, 'berkshire': 0.4,
            
            # Regulatory & Legal
            'sec': 0.6, 'cftc': 0.55, 'finra': 0.5, 'doj': 0.6, 'antitrust': 0.65,
            'regulation': 0.55, 'compliance': 0.4, 'fine': 0.5, 'settlement': 0.45,
            
            # Global Events
            'pandemic': 0.8, 'lockdown': 0.7, 'variant': 0.6, 'vaccine': 0.6,
            'climate change': 0.5, 'natural disaster': 0.6, 'cyber attack': 0.7,
            'data breach': 0.5, 'ai revolution': 0.6, 'automation': 0.5
        }
    
    def calculate_market_impact(self, text: str) -> float:
        """Enhanced market impact calculation with context awareness and keyword clustering"""
        text_lower = text.lower()
        
        # Keyword impact calculation with clustering
        keyword_impacts = []
        matched_keywords = []
        
        for keyword, weight in self.keyword_weights.items():
            if keyword in text_lower:
                keyword_impacts.append(weight)
                matched_keywords.append(keyword)
        
        if not keyword_impacts:
            return 0.15  # Slightly higher base for financial news
        
        # Context-aware adjustments
        context_multiplier = 1.0
        
        # Time sensitivity - recent/urgent language increases impact
        urgent_terms = ['breaking', 'urgent', 'just in', 'alert', 'immediate', 'emergency']
        if any(term in text_lower for term in urgent_terms):
            context_multiplier *= 1.3
        
        # Source credibility - official sources get higher weight
        official_terms = ['federal reserve', 'treasury', 'white house', 'sec announces', 'official statement']
        if any(term in text_lower for term in official_terms):
            context_multiplier *= 1.25
        
        # Market session timing (if timestamp available)
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 16:  # Market hours EST
            context_multiplier *= 1.2
        
        # Calculate weighted average impact
        avg_impact = sum(keyword_impacts) / len(keyword_impacts)
        
        # Apply diminishing returns for too many keywords (spam filter)
        keyword_count_factor = min(1.0, 5.0 / len(keyword_impacts)) if len(keyword_impacts) > 5 else 1.0
        
        # Final calculation
        final_impact = min(1.0, avg_impact * context_multiplier * keyword_count_factor)
        
        return final_impact
    
    def _analyze_trump_post(self, content: str, sentiment: Dict, market_impact: float, article: Dict) -> List[Signal]:
        """
        Enhanced specialized analyzer for posts from Donald Trump's account.
        Uses comprehensive keyword matching and market impact assessment.
        """
        signals = []
        content_lower = content.lower()

        # --- THEME 1: TRADE & TARIFFS (Enhanced with more keywords) ---
        trade_war_keywords = [
            'tariff', 'trade war', 'unfair trade', 'trade deficit', 'trade imbalance',
            'china tariffs', 'eu tariffs', 'mexico tariffs', 'canada tariffs',
            'intellectual property', 'ip theft', 'technology transfer', 'forced technology transfer',
            'dumping', 'subsidies', 'currency manipulation', 'devaluation',
            'trade deal', 'trade agreement', 'phase one', 'usmca', 'nafta',
            'wto', 'world trade organization', 'most favored nation', 'mfn',
            'protectionism', 'america first', 'buy american', 'hire american',
            'import duties', 'export controls', 'trade restrictions', 'economic sanctions'
        ]
        
        if any(word in content_lower for word in trade_war_keywords):
            # Determine severity based on specific keywords
            severity_multiplier = 1.0
            if any(severe in content_lower for severe in ['new tariffs', 'increase tariffs', '25%', '50%', 'trade war']):
                severity_multiplier = 1.2
            elif any(moderate in content_lower for severe in ['considering', 'reviewing', 'studying']):
                severity_multiplier = 0.8
                
            signals.append(Signal(
                asset='SP500', direction=Direction.BEARISH, 
                strength=self._determine_strength(0.85 * severity_multiplier, market_impact),
                confidence=0.85 * severity_multiplier, 
                reasoning=["Trade war escalation threatens corporate profits and supply chains"],
                timestamp=datetime.now(), sources=[article.get('url', '')], 
                sentiment_score=sentiment['composite'], market_impact=market_impact
            ))
            signals.append(Signal(
                asset='GOLD', direction=Direction.BULLISH, 
                strength=self._determine_strength(0.80 * severity_multiplier, market_impact),
                confidence=0.80 * severity_multiplier, 
                reasoning=["Trade uncertainty increases safe-haven gold demand"],
                timestamp=datetime.now(), sources=[article.get('url', '')], 
                sentiment_score=sentiment['composite'], market_impact=market_impact
            ))
            signals.append(Signal(
                asset='USD', direction=Direction.BEARISH, 
                strength=self._determine_strength(0.70 * severity_multiplier, market_impact),
                confidence=0.70 * severity_multiplier, 
                reasoning=["Trade tensions could trigger retaliatory currency measures"],
                timestamp=datetime.now(), sources=[article.get('url', '')], 
                sentiment_score=sentiment['composite'], market_impact=market_impact
            ))

        # --- THEME 2: FEDERAL RESERVE & MONETARY POLICY (Enhanced) ---
        fed_keywords = [
            'fed', 'federal reserve', 'powell', 'jerome powell', 'jay powell',
            'interest rates', 'rate hike', 'rate cut', 'fed funds', 'fomc',
            'monetary policy', 'quantitative easing', 'qe', 'tapering',
            'balance sheet', 'money printing', 'inflation target',
            'dot plot', 'fed meeting', 'fed minutes', 'jackson hole',
            'hawkish', 'dovish', 'tight money', 'loose money', 'easy money',
            'strong dollar', 'weak dollar', 'dollar policy', 'currency policy'
        ]
        
        if any(word in content_lower for word in fed_keywords):
            # Assess sentiment toward Fed policy
            criticism_words = ['wrong', 'mistake', 'bad', 'terrible', 'disaster', 'too high', 'too tight', 'hurting']
            praise_words = ['good', 'right', 'smart', 'correct', 'appropriate', 'helping']
            
            fed_sentiment = 0
            if any(word in content_lower for word in criticism_words):
                fed_sentiment = -0.8
            elif any(word in content_lower for word in praise_words):
                fed_sentiment = 0.8
            else:
                fed_sentiment = sentiment['composite']
                
            if fed_sentiment < -0.3:  # Criticism suggests desire for easier policy
                signals.append(Signal(
                    asset='SP500', direction=Direction.BULLISH, 
                    strength=self._determine_strength(0.75, market_impact),
                    confidence=0.75, reasoning=["Fed criticism suggests preference for easier monetary policy"],
                    timestamp=datetime.now(), sources=[article.get('url', '')], 
                    sentiment_score=sentiment['composite'], market_impact=market_impact
                ))
                signals.append(Signal(
                    asset='GOLD', direction=Direction.BULLISH, 
                    strength=self._determine_strength(0.80, market_impact),
                    confidence=0.80, reasoning=["Pressure for lower rates benefits non-yielding gold"],
                    timestamp=datetime.now(), sources=[article.get('url', '')], 
                    sentiment_score=sentiment['composite'], market_impact=market_impact
                ))
                signals.append(Signal(
                    asset='USD', direction=Direction.BEARISH, 
                    strength=self._determine_strength(0.70, market_impact),
                    confidence=0.70, reasoning=["Easier monetary policy typically weakens currency"],
                    timestamp=datetime.now(), sources=[article.get('url', '')], 
                    sentiment_score=sentiment['composite'], market_impact=market_impact
                ))

        # --- THEME 3: DOMESTIC ECONOMY & FISCAL POLICY (Enhanced) ---
        economy_keywords = [
            'economy', 'economic', 'gdp', 'growth', 'jobs', 'employment', 'unemployment',
            'tax cuts', 'tax reform', 'corporate tax', 'individual tax', 'capital gains',
            'deregulation', 'regulation', 'red tape', 'bureaucracy',
            'infrastructure', 'spending', 'stimulus', 'fiscal policy',
            'deficit', 'debt', 'budget', 'appropriations', 'congress',
            'strongest economy', 'best economy', 'greatest economy', 'record',
            'stock market', 'dow jones', 'all-time high', 'new high'
        ]
        
        positive_economy_words = ['strongest', 'best', 'greatest', 'record', 'amazing', 'incredible', 'fantastic']
        negative_economy_words = ['terrible', 'worst', 'disaster', 'failing', 'broken', 'crisis']
        
        if any(word in content_lower for word in economy_keywords):
            economy_sentiment = sentiment['composite']
            if any(word in content_lower for word in positive_economy_words):
                economy_sentiment = max(economy_sentiment, 0.8)
            elif any(word in content_lower for word in negative_economy_words):
                economy_sentiment = min(economy_sentiment, -0.8)
                
            if economy_sentiment > 0.3:
                signals.append(Signal(
                    asset='SP500', direction=Direction.BULLISH, 
                    strength=self._determine_strength(0.80, market_impact),
                    confidence=0.85, reasoning=["Positive economic commentary supports market confidence"],
                    timestamp=datetime.now(), sources=[article.get('url', '')], 
                    sentiment_score=sentiment['composite'], market_impact=market_impact
                ))
            elif economy_sentiment < -0.3:
                signals.append(Signal(
                    asset='SP500', direction=Direction.BEARISH, 
                    strength=self._determine_strength(0.75, market_impact),
                    confidence=0.80, reasoning=["Negative economic commentary creates market uncertainty"],
                    timestamp=datetime.now(), sources=[article.get('url', '')], 
                    sentiment_score=sentiment['composite'], market_impact=market_impact
                ))

        # --- THEME 4: CRYPTOCURRENCY & BITCOIN (Enhanced) ---
        crypto_keywords = [
            'bitcoin', 'btc', 'cryptocurrency', 'crypto', 'digital currency', 'digital asset',
            'blockchain', 'mining', 'miners', 'hash rate', 'proof of work',
            'ethereum', 'eth', 'altcoin', 'defi', 'decentralized finance',
            'nft', 'non-fungible token', 'web3', 'metaverse',
            'coinbase', 'binance', 'crypto exchange', 'crypto regulation',
            'sec crypto', 'cftc crypto', 'treasury crypto', 'irs crypto',
            'cbdc', 'central bank digital currency', 'fed coin', 'digital dollar',
            'stablecoin', 'usdc', 'usdt', 'tether', 'stable value',
            'el salvador', 'legal tender', 'adoption', 'mainstream adoption'
        ]
        
        crypto_positive = ['legal tender', 'adoption', 'innovation', 'future', 'revolutionary', 'freedom', 'decentralized']
        crypto_negative = ['scam', 'fraud', 'ponzi', 'bubble', 'speculation', 'ban', 'illegal', 'criminal']
        
        if any(word in content_lower for word in crypto_keywords):
            crypto_sentiment = sentiment['composite']
            if any(word in content_lower for word in crypto_positive):
                crypto_sentiment = max(crypto_sentiment, 0.7)
            elif any(word in content_lower for word in crypto_negative):
                crypto_sentiment = min(crypto_sentiment, -0.7)
                
            direction = Direction.BULLISH if crypto_sentiment >= 0 else Direction.BEARISH
            reasoning_text = ("Positive cryptocurrency sentiment from influential figure" if direction == Direction.BULLISH 
                            else "Negative cryptocurrency sentiment from influential figure")
            
            signals.append(Signal(
                asset='BITCOIN', direction=direction, 
                strength=self._determine_strength(0.85, market_impact),
                confidence=0.85, reasoning=[reasoning_text],
                timestamp=datetime.now(), sources=[article.get('url', '')], 
                sentiment_score=sentiment['composite'], market_impact=market_impact
            ))

        # --- THEME 5: GEOPOLITICAL & FOREIGN POLICY (Enhanced) ---
        geopolitical_keywords = [
            'china', 'chinese', 'xi jinping', 'ccp', 'beijing', 'hong kong', 'taiwan',
            'russia', 'russian', 'putin', 'moscow', 'ukraine', 'crimea',
            'iran', 'iranian', 'middle east', 'nuclear', 'sanctions',
            'north korea', 'kim jong', 'pyongyang', 'nuclear weapons',
            'nato', 'european union', 'eu', 'brexit', 'germany', 'france',
            'israel', 'palestine', 'syria', 'afghanistan', 'iraq',
            'military', 'defense', 'war', 'conflict', 'peace', 'treaty',
            'allies', 'alliance', 'partnership', 'diplomacy', 'summit',
            'trade partner', 'economic partner', 'strategic partner'
        ]
        
        conflict_words = ['war', 'conflict', 'tension', 'crisis', 'threat', 'hostile', 'enemy']
        cooperation_words = ['deal', 'agreement', 'cooperation', 'partnership', 'alliance', 'friend']
        
        if any(word in content_lower for word in geopolitical_keywords):
            geo_sentiment = sentiment['composite']
            if any(word in content_lower for word in conflict_words):
                geo_sentiment = min(geo_sentiment, -0.5)
            elif any(word in content_lower for word in cooperation_words):
                geo_sentiment = max(geo_sentiment, 0.5)
                
            if geo_sentiment < -0.2:  # Negative geopolitical sentiment
                signals.append(Signal(
                    asset='GOLD', direction=Direction.BULLISH, 
                    strength=self._determine_strength(0.80, market_impact),
                    confidence=0.80, reasoning=["Geopolitical tensions increase safe-haven demand"],
                    timestamp=datetime.now(), sources=[article.get('url', '')], 
                    sentiment_score=sentiment['composite'], market_impact=market_impact
                ))
                signals.append(Signal(
                    asset='SP500', direction=Direction.BEARISH, 
                    strength=self._determine_strength(0.70, market_impact),
                    confidence=0.75, reasoning=["Geopolitical uncertainty creates market volatility"],
                    timestamp=datetime.now(), sources=[article.get('url', '')], 
                    sentiment_score=sentiment['composite'], market_impact=market_impact
                ))
                
                # USD can be both safe haven and hurt by trade tensions
                if 'china' in content_lower or 'trade' in content_lower:
                    signals.append(Signal(
                        asset='USD', direction=Direction.BEARISH, 
                        strength=self._determine_strength(0.65, market_impact),
                        confidence=0.65, reasoning=["Trade tensions may hurt dollar through reduced global trade"],
                        timestamp=datetime.now(), sources=[article.get('url', '')], 
                        sentiment_score=sentiment['composite'], market_impact=market_impact
                    ))
                else:
                    signals.append(Signal(
                        asset='USD', direction=Direction.BULLISH, 
                        strength=self._determine_strength(0.70, market_impact),
                        confidence=0.70, reasoning=["USD benefits as global safe haven during tensions"],
                        timestamp=datetime.now(), sources=[article.get('url', '')], 
                        sentiment_score=sentiment['composite'], market_impact=market_impact
                    ))

        return signals
    
    def generate_signals(self, article: Dict) -> List[Signal]:
        """Enhanced signal generation with improved filtering and validation"""
        title = article.get('title', '')
        description = article.get('description', '')
        content = f"{title}. {description}"
        source = article.get('source', {})
        source_name = source.get('name')
        source_user = source.get('user')

        # Content quality check - filter out low-quality articles
        if len(content.strip()) < 20:
            return []
        
        # Check for duplicate/spam content
        spam_indicators = ['click here', 'subscribe now', 'limited time', '!!!!!']
        if any(indicator in content.lower() for indicator in spam_indicators):
            return []

        # Enhanced sentiment analysis
        sentiment_data = self.sentiment_analyzer.analyze_sentiment(content)
        
        # Calculate market impact
        market_impact = self.calculate_market_impact(content)
        
        # Early exit if very low impact and weak sentiment
        if market_impact < 0.2 and sentiment_data['confidence'] < 0.3:
            return []

        # Trump post special handling
        if source_name == 'Truth Social' and source_user == 'realDonaldTrump':
            print("✅ Detected post from Trump. Applying specialized analysis.")
            market_impact = max(market_impact, 0.85)  # Ensure minimum high impact
            return self._analyze_trump_post(content, sentiment_data, market_impact, article)

        # General signal generation with improved filtering
        signals = []
        for asset_name, asset_config in Config.ASSETS.items():
            signal = self._analyze_asset_impact(
                asset_name, asset_config, content,
                sentiment_data, market_impact, article
            )
            if signal and signal.confidence > 0.4:  # Minimum confidence threshold
                signals.append(signal)

        # Remove conflicting signals (same asset, opposite directions, similar timestamps)
        signals = self._resolve_signal_conflicts(signals)
        
        return signals
    
    def _analyze_asset_impact(self, asset_name: str, asset_config: Dict, 
                        content: str, sentiment_data: Dict, 
                        market_impact: float, article: Dict) -> Optional[Signal]:
        """Enhanced asset impact analysis with improved keyword matching and context"""
        content_lower = content.lower()
        
        # Multi-level keyword relevance check
        keyword_matches = 0
        keyword_strength = 0
        
        for keyword in asset_config['keywords']:
            if keyword in content_lower:
                keyword_matches += 1
                # Weight longer, more specific keywords higher
                keyword_strength += len(keyword.split())
        
        # Enhanced relevance threshold
        relevance_score = (keyword_matches + keyword_strength * 0.5) * asset_config['weight']
        
        # Skip if not relevant enough, unless high market impact
        if relevance_score < 1.0 and market_impact < 0.6:
            return None
        
        # Asset-specific analysis with improved context
        direction = Direction.NEUTRAL
        reasoning = []
        base_confidence = sentiment_data['confidence'] * asset_config['weight']
        
        # Apply asset-specific analysis
        if asset_name == 'GOLD':
            direction, reasoning = self._analyze_gold(content_lower, sentiment_data)
        elif asset_name == 'BITCOIN':
            direction, reasoning = self._analyze_bitcoin(content_lower, sentiment_data)
        elif asset_name == 'SP500':
            direction, reasoning = self._analyze_sp500(content_lower, sentiment_data)
        elif asset_name == 'USD':
            direction, reasoning = self._analyze_usd(content_lower, sentiment_data)
        
        # Enhance confidence based on reasoning strength
        reasoning_bonus = min(0.2, len(reasoning) * 0.05)
        final_confidence = min(0.95, base_confidence + reasoning_bonus)
        
        # Determine signal strength
        strength = self._determine_strength(final_confidence, market_impact, relevance_score)
        
        # Enhanced sentiment score alignment
        final_sentiment_score = self._align_sentiment_with_direction(
            sentiment_data['composite'], direction, final_confidence
        )
        
        # Filter out weak signals unless high market impact
        if direction == Direction.NEUTRAL or (strength == SignalStrength.WEAK and market_impact < 0.5):
            return None
        
        return Signal(
            asset=asset_name,
            direction=direction,
            strength=strength,
            confidence=final_confidence,
            reasoning=reasoning,
            timestamp=datetime.now(),
            sources=[article.get('url', '')],
            sentiment_score=final_sentiment_score,
            market_impact=market_impact
        )
    
    def _analyze_gold(self, content: str, sentiment: Dict) -> Tuple[Direction, List[str]]:
        """Enhanced Gold-specific analysis with comprehensive keyword coverage"""
        reasoning = []
        direction = Direction.NEUTRAL
        content_lower = content.lower()
        
        # BULLISH FACTORS FOR GOLD
        
        # Inflation & Currency Debasement (High Impact)
        inflation_keywords = ['inflation', 'cpi', 'core cpi', 'ppi', 'pce', 'core pce', 
                            'price increases', 'cost of living', 'purchasing power',
                            'money printing', 'quantitative easing', 'qe', 'currency debasement',
                            'fiscal stimulus', 'government spending', 'deficit spending']
        
        if any(keyword in content_lower for keyword in inflation_keywords):
            inflation_indicators = ['rising', 'increasing', 'surge', 'spike', 'above target', 'elevated', 'persistent']
            if any(indicator in content_lower for indicator in inflation_indicators) or sentiment['composite'] > 0.2:
                direction = Direction.BULLISH
                reasoning.append("Rising inflation increases gold's appeal as inflation hedge")
        
        # Safe Haven Demand (High Impact)
        safe_haven_keywords = ['geopolitical tension', 'war', 'conflict', 'crisis', 'uncertainty',
                            'market volatility', 'stock market crash', 'recession fears', 
                            'economic instability', 'banking crisis', 'financial crisis',
                            'trade war', 'sanctions', 'political instability', 'election uncertainty',
                            'brexit', 'sovereign debt', 'debt crisis', 'currency crisis']
        
        if any(keyword in content_lower for keyword in safe_haven_keywords):
            direction = Direction.BULLISH
            reasoning.append("Crisis conditions drive safe-haven demand for gold")
        
        # Central Bank Policy (Variable Impact)
        dovish_keywords = ['dovish', 'accommodative', 'easy money', 'low rates', 'rate cut',
                        'monetary stimulus', 'negative rates', 'yield curve control']
        
        if any(keyword in content_lower for keyword in dovish_keywords):
            direction = Direction.BULLISH
            reasoning.append("Accommodative monetary policy reduces opportunity cost of holding gold")
        
        # Dollar Weakness (Moderate Impact)
        weak_dollar_keywords = ['weak dollar', 'dollar decline', 'dollar weakness', 'dxy falling',
                            'currency devaluation', 'competitive devaluation']
        
        if any(keyword in content_lower for keyword in weak_dollar_keywords):
            direction = Direction.BULLISH
            reasoning.append("Dollar weakness makes gold cheaper for international buyers")
        
        # Central Bank Buying (Moderate Impact)
        cb_buying_keywords = ['central bank buying', 'reserve diversification', 'gold reserves',
                            'official sector demand', 'sovereign buying']
        
        if any(keyword in content_lower for keyword in cb_buying_keywords):
            direction = Direction.BULLISH
            reasoning.append("Central bank gold purchases support price")
        
        # BEARISH FACTORS FOR GOLD
        
        # Rising Interest Rates (High Impact)
        hawkish_keywords = ['hawkish', 'tightening', 'rate hike', 'higher rates', 'restrictive policy',
                        'fed tightening', 'monetary tightening', 'normalize rates', 'rate increase']
        
        if any(keyword in content_lower for keyword in hawkish_keywords):
            if direction != Direction.BULLISH:  # Don't override bullish signals
                direction = Direction.BEARISH
                reasoning.append("Rising interest rates increase opportunity cost of holding non-yielding gold")
        
        # Strong Dollar (High Impact)
        strong_dollar_keywords = ['strong dollar', 'dollar strength', 'dxy rising', 'dollar rally',
                                'dollar appreciation', 'dollar index up']
        
        if any(keyword in content_lower for keyword in strong_dollar_keywords):
            if direction != Direction.BULLISH:
                direction = Direction.BEARISH
                reasoning.append("Strong dollar makes gold more expensive for foreign buyers")
        
        # Risk-On Sentiment (Moderate Impact)
        risk_on_keywords = ['risk appetite', 'risk-on', 'market optimism', 'equity rally',
                        'growth optimism', 'economic recovery', 'market confidence']
        
        if any(keyword in content_lower for keyword in risk_on_keywords):
            if direction != Direction.BULLISH and sentiment['composite'] > 0.3:
                direction = Direction.BEARISH
                reasoning.append("Risk-on sentiment reduces safe-haven demand for gold")
        
        # Technical Factors
        technical_bearish = ['gold sell-off', 'gold decline', 'gold weakness', 'support broken',
                            'technical breakdown', 'selling pressure']
        
        if any(keyword in content_lower for keyword in technical_bearish):
            if direction != Direction.BULLISH:
                direction = Direction.BEARISH
                reasoning.append("Technical selling pressure weighs on gold prices")
        
        return direction, reasoning
    
    def _analyze_bitcoin(self, content: str, sentiment: Dict) -> Tuple[Direction, List[str]]:
        """Enhanced Bitcoin-specific analysis with comprehensive coverage"""
        reasoning = []
        direction = Direction.NEUTRAL
        content_lower = content.lower()
        
        # BULLISH FACTORS FOR BITCOIN
        
        # Institutional Adoption (High Impact)
        institutional_keywords = ['institutional adoption', 'corporate treasury', 'bitcoin etf', 'etf approval',
                                'wall street', 'goldman sachs', 'jp morgan', 'morgan stanley', 'blackrock',
                                'fidelity', 'grayscale', 'microstrategy', 'tesla bitcoin', 'corporate buying',
                                'pension fund', 'endowment', 'sovereign wealth fund', 'family office']
        
        if any(keyword in content_lower for keyword in institutional_keywords):
            if sentiment['composite'] > 0.1:
                direction = Direction.BULLISH
                reasoning.append("Institutional adoption increases Bitcoin legitimacy and demand")
        
        # Positive Regulatory Development (High Impact)
        positive_reg_keywords = ['regulatory clarity', 'crypto regulation', 'legal framework', 'bitcoin legal',
                                'sec approval', 'cftc commodity', 'legal tender', 'regulatory approval',
                                'compliance framework', 'institutional custody', 'regulated exchange']
        
        if any(keyword in content_lower for keyword in positive_reg_keywords):
            if sentiment['composite'] > 0.2:
                direction = Direction.BULLISH
                reasoning.append("Positive regulatory developments provide clarity and institutional confidence")
        
        # Technological Progress (Moderate Impact)
        tech_keywords = ['lightning network', 'taproot', 'bitcoin upgrade', 'scaling solution',
                        'layer 2', 'bitcoin mining', 'hash rate', 'network security', 'decentralization',
                        'bitcoin development', 'protocol improvement', 'bitcoin core']
        
        if any(keyword in content_lower for keyword in tech_keywords):
            if sentiment['composite'] > 0.2:
                direction = Direction.BULLISH
                reasoning.append("Technological improvements enhance Bitcoin's utility and adoption")
        
        # Inflation Hedge Narrative (Moderate Impact)
        inflation_hedge_keywords = ['digital gold', 'store of value', 'inflation hedge', 'currency debasement',
                                'monetary inflation', 'bitcoin vs gold', 'hard asset', 'scarce asset',
                                'fixed supply', '21 million', 'halving', 'stock to flow']
        
        if any(keyword in content_lower for keyword in inflation_hedge_keywords):
            if sentiment['composite'] > 0.1:
                direction = Direction.BULLISH
                reasoning.append("Bitcoin gaining acceptance as digital store of value and inflation hedge")
        
        # Network Growth (Moderate Impact)
        network_keywords = ['bitcoin adoption', 'user growth', 'wallet addresses', 'transaction volume',
                        'payment adoption', 'merchant acceptance', 'bitcoin payments', 'on-chain metrics',
                        'active addresses', 'bitcoin usage']
        
        if any(keyword in content_lower for keyword in network_keywords):
            if sentiment['composite'] > 0.2:
                direction = Direction.BULLISH
                reasoning.append("Growing network usage demonstrates increasing utility")
        
        # BEARISH FACTORS FOR BITCOIN
        
        # Regulatory Crackdown (High Impact)
        negative_reg_keywords = ['crypto ban', 'bitcoin ban', 'mining ban', 'exchange ban', 'regulatory crackdown',
                                'sec lawsuit', 'cftc action', 'doj investigation', 'illegal', 'criminal activity',
                                'aml concerns', 'kyc requirements', 'tax evasion', 'money laundering']
        
        if any(keyword in content_lower for keyword in negative_reg_keywords):
            if sentiment['composite'] < -0.1:
                direction = Direction.BEARISH
                reasoning.append("Regulatory crackdowns create uncertainty and restrict access")
        
        # Environmental Concerns (Moderate Impact)
        environmental_keywords = ['bitcoin energy', 'mining energy', 'carbon footprint', 'environmental impact',
                                'energy consumption', 'fossil fuels', 'renewable energy', 'green mining',
                                'sustainability', 'esg concerns', 'climate change', 'carbon emissions']
        
        if any(keyword in content_lower for keyword in environmental_keywords):
            if sentiment['composite'] < -0.2:
                direction = Direction.BEARISH
                reasoning.append("Environmental concerns create regulatory and adoption headwinds")
        
        # Market Risk-Off Sentiment (High Impact)
        risk_off_keywords = ['risk-off', 'market crash', 'liquidity crisis', 'margin calls', 'deleveraging',
                            'crypto winter', 'bear market', 'sell-off', 'flight to safety', 'risk aversion',
                            'correlation breakdown', 'crypto contagion']
        
        if any(keyword in content_lower for keyword in risk_off_keywords):
            direction = Direction.BEARISH
            reasoning.append("Risk-off sentiment pressures speculative assets like Bitcoin")
        
        # Exchange/Infrastructure Issues (High Impact)
        infrastructure_keywords = ['exchange hack', 'wallet hack', 'crypto theft', 'exchange collapse',
                                'ftx', 'celsius', 'luna', 'ust', 'stablecoin depeg', 'crypto fraud',
                                'ponzi scheme', 'rug pull', 'smart contract exploit']
        
        if any(keyword in content_lower for keyword in infrastructure_keywords):
            if sentiment['composite'] < -0.2:
                direction = Direction.BEARISH
                reasoning.append("Infrastructure failures and fraud damage crypto ecosystem confidence")
        
        # Technical Breakdown (Moderate Impact)
        technical_bearish = ['bitcoin breakdown', 'support broken', 'death cross', 'technical sell-off',
                            'whale selling', 'miner selling', 'long liquidation', 'funding negative']
        
        if any(keyword in content_lower for keyword in technical_bearish):
            if direction != Direction.BULLISH:
                direction = Direction.BEARISH
                reasoning.append("Technical breakdown triggers algorithmic and momentum selling")
        
        # Interest Rate Impact (Moderate Impact)
        rates_keywords = ['interest rates', 'fed hikes', 'monetary tightening', 'liquidity drain',
                        'risk-free rate', 'opportunity cost', 'yield competition']
        
        if any(keyword in content_lower for keyword in rates_keywords):
            if sentiment['composite'] < -0.1:
                if direction != Direction.BULLISH:
                    direction = Direction.BEARISH
                    reasoning.append("Rising rates increase opportunity cost of holding non-yielding Bitcoin")
        
        return direction, reasoning

    def _analyze_sp500(self, content: str, sentiment: Dict) -> Tuple[Direction, List[str]]:
        """Enhanced S&P 500 specific analysis with comprehensive coverage"""
        reasoning = []
        direction = Direction.NEUTRAL
        content_lower = content.lower()
        
        # BULLISH FACTORS FOR S&P 500
        
        # Economic Growth (High Impact)
        growth_keywords = ['gdp growth', 'economic expansion', 'robust growth', 'strong economy',
                        'consumer spending', 'business investment', 'capex', 'productivity growth',
                        'employment growth', 'wage growth', 'economic recovery', 'v-shaped recovery']
        
        if any(keyword in content_lower for keyword in growth_keywords):
            if sentiment['composite'] > 0.2:
                direction = Direction.BULLISH
                reasoning.append("Strong economic growth supports corporate earnings and valuations")
        
        # Federal Reserve Policy (Variable Impact)
        dovish_fed_keywords = ['dovish fed', 'accommodative policy', 'easy money', 'rate cuts', 'fed pivot',
                            'pause rate hikes', 'lower for longer', 'quantitative easing', 'balance sheet expansion',
                            'fed put', 'powell dovish', 'soft landing']
        
        if any(keyword in content_lower for keyword in dovish_fed_keywords):
            direction = Direction.BULLISH
            reasoning.append("Accommodative Fed policy supports risk asset valuations")
        
        # Corporate Earnings (High Impact)
        earnings_keywords = ['strong earnings', 'earnings beat', 'revenue growth', 'profit margins',
                            'earnings guidance', 'eps growth', 'record profits', 'margin expansion',
                            'operating leverage', 'cost control', 'pricing power', 'earnings season']
        
        if any(keyword in content_lower for keyword in earnings_keywords):
            if sentiment['composite'] > 0.3:
                direction = Direction.BULLISH
                reasoning.append("Strong corporate fundamentals justify higher equity valuations")
        
        # Market Technicals (Moderate Impact)
        technical_bullish = ['breakout', 'new highs', 'all-time high', 'momentum', 'bullish pattern',
                            'support holding', 'golden cross', 'trend intact', 'buying pressure',
                            'institutional buying', 'retail inflows', 'etf inflows']
        
        if any(keyword in content_lower for keyword in technical_bullish):
            if sentiment['composite'] > 0.2:
                direction = Direction.BULLISH
                reasoning.append("Positive technical signals and fund flows support continued uptrend")
        
        # Sector Rotation (Moderate Impact)
        sector_strength = ['tech rally', 'growth stocks', 'innovation', 'ai revolution', 'digital transformation',
                        'cloud computing', 'semiconductor strength', 'biotech breakthrough', 'energy transition',
                        'infrastructure spending', 'reshoring', 'nearshoring']
        
        if any(keyword in content_lower for keyword in sector_strength):
            if sentiment['composite'] > 0.2:
                direction = Direction.BULLISH
                reasoning.append("Sector strength and thematic trends drive market leadership")
        
        # BEARISH FACTORS FOR S&P 500
        
        # Economic Recession (High Impact)
        recession_keywords = ['recession', 'economic contraction', 'gdp decline', 'negative growth',
                            'hard landing', 'economic slowdown', 'consumer weakness', 'spending decline',
                            'layoffs', 'unemployment rising', 'jobless claims', 'leading indicators']
        
        if any(keyword in content_lower for keyword in recession_keywords):
            if sentiment['composite'] < -0.2:
                direction = Direction.BEARISH
                reasoning.append("Recession fears and economic weakness pressure corporate earnings")
        
        # Federal Reserve Hawkishness (High Impact)
        hawkish_fed_keywords = ['hawkish fed', 'aggressive tightening', 'rate hikes', 'higher for longer',
                            'quantitative tightening', 'qt', 'balance sheet reduction', 'restrictive policy',
                            'powell hawkish', 'fed tightening', 'monetary restraint', 'liquidity drain']
        
        if any(keyword in content_lower for keyword in hawkish_fed_keywords):
            direction = Direction.BEARISH
            reasoning.append("Hawkish Fed policy increases discount rates and pressures valuations")
        
        # Inflation Concerns (Moderate Impact)
        inflation_keywords = ['inflation surge', 'price pressures', 'cost inflation', 'margin compression',
                            'input costs', 'wage inflation', 'cpi spike', 'pce elevated', 'pricing power',
                            'pass-through', 'inflation expectations']
        
        if any(keyword in content_lower for keyword in inflation_keywords):
            if sentiment['composite'] < -0.1:
                if direction != Direction.BULLISH:
                    direction = Direction.BEARISH
                    reasoning.append("Inflation pressures corporate margins and may force Fed tightening")
        
        # Earnings Concerns (High Impact)
        earnings_concerns = ['earnings miss', 'guidance cut', 'margin pressure', 'revenue decline',
                            'profit warning', 'earnings recession', 'multiple compression', 'valuation reset',
                            'eps decline', 'cost pressures', 'demand destruction']
        
        if any(keyword in content_lower for keyword in earnings_concerns):
            if sentiment['composite'] < -0.2:
                direction = Direction.BEARISH
                reasoning.append("Earnings deterioration undermines equity valuations")
        
        # Geopolitical Risks (Moderate Impact)
        geopolitical_keywords = ['trade war', 'geopolitical tension', 'supply chain', 'tariffs', 'sanctions',
                                'war', 'conflict', 'energy crisis', 'commodity shock', 'deglobalization',
                                'china tensions', 'taiwan', 'middle east', 'russia ukraine']
        
        if any(keyword in content_lower for keyword in geopolitical_keywords):
            if sentiment['composite'] < -0.2:
                if direction != Direction.BULLISH:
                    direction = Direction.BEARISH
                    reasoning.append("Geopolitical risks create uncertainty and disrupt business operations")
        
        # Technical Breakdown (Moderate Impact)
        technical_bearish = ['breakdown', 'support broken', 'death cross', 'selling pressure', 'distribution',
                            'insider selling', 'margin calls', 'forced selling', 'volatility spike', 'vix surge',
                            'risk-off', 'flight to quality']
        
        if any(keyword in content_lower for keyword in technical_bearish):
            if direction != Direction.BULLISH:
                direction = Direction.BEARISH
                reasoning.append("Technical breakdown and risk-off sentiment pressure equity markets")
        
        # Banking/Financial System (High Impact)
        financial_stress = ['banking crisis', 'credit crunch', 'loan losses', 'financial instability',
                        'bank failures', 'liquidity crisis', 'credit tightening', 'yield curve inversion',
                        'term premium', 'credit spreads', 'systemic risk']
        
        if any(keyword in content_lower for keyword in financial_stress):
            if sentiment['composite'] < -0.3:
                direction = Direction.BEARISH
                reasoning.append("Financial system stress threatens economic growth and market stability")
        
        return direction, reasoning

    def _analyze_usd(self, content: str, sentiment: Dict) -> Tuple[Direction, List[str]]:
        """Enhanced USD specific analysis with comprehensive coverage"""
        reasoning = []
        direction = Direction.NEUTRAL
        content_lower = content.lower()
        
        # BULLISH FACTORS FOR USD
        
        # Federal Reserve Hawkishness (High Impact)
        hawkish_fed_keywords = ['hawkish fed', 'rate hikes', 'aggressive tightening', 'restrictive policy',
                            'higher for longer', 'fed tightening', 'monetary restraint', 'quantitative tightening',
                            'powell hawkish', 'fed funds rate', 'terminal rate', 'neutral rate']
        
        if any(keyword in content_lower for keyword in hawkish_fed_keywords):
            if sentiment['composite'] > -0.1:  # Not overly negative
                direction = Direction.BULLISH
                reasoning.append("Hawkish Fed policy attracts capital flows and strengthens dollar")
        
        # US Economic Outperformance (High Impact)
        economic_strength = ['us growth', 'gdp outperformance', 'economic resilience', 'consumer strength',
                            'labor market', 'job growth', 'productivity', 'competitiveness', 'innovation',
                            'energy independence', 'manufacturing renaissance', 'reshoring']
        
        if any(keyword in content_lower for keyword in economic_strength):
            if sentiment['composite'] > 0.2:
                direction = Direction.BULLISH
                reasoning.append("US economic outperformance supports dollar strength")
        
        # Safe Haven Demand (High Impact)
        safe_haven_keywords = ['safe haven', 'flight to quality', 'risk-off', 'global uncertainty',
                            'geopolitical tension', 'market volatility', 'financial crisis', 'recession fears',
                            'banking crisis', 'sovereign debt', 'emerging market crisis', 'capital flight']
        
        if any(keyword in content_lower for keyword in safe_haven_keywords):
            direction = Direction.BULLISH
            reasoning.append("Safe haven flows during uncertainty strengthen the dollar")
        
        # Interest Rate Differential (High Impact)
        rate_differential = ['yield advantage', 'rate differential', 'carry trade', 'real rates',
                            'nominal rates', 'interest rate parity', 'yield curve', 'term premium',
                            'bond yields', 'treasury yields', 'funding costs']
        
        if any(keyword in content_lower for keyword in rate_differential):
            if sentiment['composite'] > 0.1:
                direction = Direction.BULLISH
                reasoning.append("Favorable interest rate differentials attract foreign investment")
        
        # Strong Employment Data (Moderate Impact)
        employment_keywords = ['nonfarm payrolls', 'nfp', 'job creation', 'unemployment rate', 'jobless claims',
                            'labor participation', 'wage growth', 'job openings', 'quits rate', 'hiring',
                            'employment cost index', 'labor shortage']
        
        if any(keyword in content_lower for keyword in employment_keywords):
            if sentiment['composite'] > 0.2:
                direction = Direction.BULLISH
                reasoning.append("Strong employment data supports Fed tightening expectations and USD")
        
        # BEARISH FACTORS FOR USD
        
        # Federal Reserve Dovishness (High Impact)
        dovish_fed_keywords = ['dovish fed', 'accommodative policy', 'rate cuts', 'fed pivot', 'pause',
                            'lower for longer', 'quantitative easing', 'balance sheet expansion', 'fed put',
                            'soft landing', 'growth concerns', 'employment mandate']
        
        if any(keyword in content_lower for keyword in dovish_fed_keywords):
            direction = Direction.BEARISH
            reasoning.append("Dovish Fed policy reduces yield advantage and weakens dollar")
        
        # US Economic Weakness (High Impact)
        economic_weakness = ['us recession', 'gdp contraction', 'economic slowdown', 'consumer weakness',
                            'business investment decline', 'productivity decline', 'competitiveness loss',
                            'trade deficit', 'current account deficit', 'twin deficits']
        
        if any(keyword in content_lower for keyword in economic_weakness):
            if sentiment['composite'] < -0.2:
                direction = Direction.BEARISH
                reasoning.append("US economic weakness undermines dollar fundamentals")
        
        # Fiscal Concerns (Moderate Impact)
        fiscal_keywords = ['budget deficit', 'government debt', 'debt ceiling', 'fiscal sustainability',
                        'debt-to-gdp', 'crowding out', 'fiscal dominance', 'unfunded liabilities',
                        'social security', 'medicare', 'infrastructure spending']
        
        if any(keyword in content_lower for keyword in fiscal_keywords):
            if sentiment['composite'] < -0.2:
                if direction != Direction.BULLISH:
                    direction = Direction.BEARISH
                    reasoning.append("Fiscal concerns raise questions about long-term dollar stability")
        
        # International Developments (Variable Impact)
        international_keywords = ['dedollarization', 'reserve currency', 'swift alternatives', 'digital currencies',
                                'cbdc', 'yuan internationalization', 'brics currency', 'commodity currencies',
                                'petrodollar', 'euro strength', 'yen intervention']
        
        if any(keyword in content_lower for keyword in international_keywords):
            if sentiment['composite'] < -0.1:
                if direction != Direction.BULLISH:
                    direction = Direction.BEARISH
                    reasoning.append("Challenges to dollar's international role create headwinds")
        
        # Trade and Current Account (Moderate Impact)
        trade_keywords = ['trade deficit', 'current account', 'export competitiveness', 'import surge',
                        'trade balance', 'goods deficit', 'services surplus', 'investment flows',
                        'capital account', 'balance of payments']
        
        if any(keyword in content_lower for keyword in trade_keywords):
            if sentiment['composite'] < -0.2:
                if direction != Direction.BULLISH:
                    direction = Direction.BEARISH
                    reasoning.append("Trade and current account deficits create dollar selling pressure")
        
        # Technical Factors (Moderate Impact)
        technical_bearish = ['dollar breakdown', 'dxy decline', 'support broken', 'trend reversal',
                            'technical selling', 'momentum loss', 'overbought correction', 'profit taking']
        
        if any(keyword in content_lower for keyword in technical_bearish):
            if direction != Direction.BULLISH:
                direction = Direction.BEARISH
                reasoning.append("Technical breakdown triggers algorithmic and momentum selling")
        
        # Energy Prices (Variable Impact - Complex relationship)
        energy_keywords = ['oil prices', 'energy crisis', 'gas prices', 'commodity prices', 'inflation',
                        'energy independence', 'shale production', 'strategic reserve', 'opec']
        
        if any(keyword in content_lower for keyword in energy_keywords):
            if 'energy independence' in content_lower or 'shale production' in content_lower:
                if sentiment['composite'] > 0.1 and direction != Direction.BEARISH:
                    direction = Direction.BULLISH
                    reasoning.append("US energy independence strengthens dollar fundamentals")
            elif sentiment['composite'] < -0.2 and 'crisis' in content_lower:
                if direction != Direction.BULLISH:
                    direction = Direction.BEARISH
                    reasoning.append("Energy crisis creates inflationary pressures and economic headwinds")
        
        return direction, reasoning
    
    def _determine_strength(self, confidence: float, market_impact: float, relevance_score: float = 1.0) -> SignalStrength:
        """Enhanced strength determination with multiple factors"""
        # Weighted combination of factors
        combined_score = (
            confidence * 0.4 +
            market_impact * 0.5 +
            min(relevance_score / 3.0, 0.3) * 0.1  # Cap relevance contribution
        )
        
        # Dynamic thresholds based on market conditions
        # You could add volatility index or market session adjustments here
        base_thresholds = {
            'critical': 0.75,
            'strong': 0.55,
            'moderate': 0.35,
            'weak': 0.0
        }
        
        if combined_score >= base_thresholds['critical']:
            return SignalStrength.CRITICAL
        elif combined_score >= base_thresholds['strong']:
            return SignalStrength.STRONG
        elif combined_score >= base_thresholds['moderate']:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
        
    def _align_sentiment_with_direction(self, original_sentiment: float, direction: Direction, confidence: float) -> float:
        """Align sentiment score with determined direction based on analysis confidence"""
        if direction == Direction.NEUTRAL:
            return original_sentiment
        
        # If we're very confident in our directional analysis, adjust sentiment accordingly
        if confidence > 0.7:
            if direction == Direction.BULLISH and original_sentiment < 0:
                # Our analysis says bullish despite negative sentiment - weight our analysis more
                return abs(original_sentiment) * 0.8
            elif direction == Direction.BEARISH and original_sentiment > 0:
                # Our analysis says bearish despite positive sentiment
                return -abs(original_sentiment) * 0.8
        
        # For lower confidence, blend original sentiment with direction
        direction_weight = confidence * 0.5  # Max 50% influence
        sentiment_weight = 1 - direction_weight
        
        target_sentiment = 0.7 if direction == Direction.BULLISH else -0.7
        return original_sentiment * sentiment_weight + target_sentiment * direction_weight

    def _resolve_signal_conflicts(self, signals: List[Signal]) -> List[Signal]:
        """Resolve conflicting signals for the same asset"""
        if len(signals) <= 1:
            return signals
        
        # Group signals by asset
        asset_signals = {}
        for signal in signals:
            if signal.asset not in asset_signals:
                asset_signals[signal.asset] = []
            asset_signals[signal.asset].append(signal)
        
        resolved_signals = []
        
        for asset, signal_list in asset_signals.items():
            if len(signal_list) == 1:
                resolved_signals.extend(signal_list)
                continue
            
            # Check for conflicts (opposite directions)
            bullish_signals = [s for s in signal_list if s.direction == Direction.BULLISH]
            bearish_signals = [s for s in signal_list if s.direction == Direction.BEARISH]
            
            if bullish_signals and bearish_signals:
                # Conflict detected - keep the highest confidence signal
                all_directional = bullish_signals + bearish_signals
                best_signal = max(all_directional, key=lambda s: s.confidence * s.market_impact)
                
                # Combine reasoning from conflicting signals
                all_reasoning = []
                for signal in all_directional:
                    all_reasoning.extend(signal.reasoning)
                
                best_signal.reasoning = list(set(all_reasoning))  # Remove duplicates
                resolved_signals.append(best_signal)
            else:
                # No conflict - keep all signals or merge similar ones
                resolved_signals.extend(signal_list)
        
        return resolved_signals

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
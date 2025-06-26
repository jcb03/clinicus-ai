from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class TextAnalyzer:
    def __init__(self):
        """Initialize text analysis models"""
        try:
            # Load sentiment analysis model
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            # Load emotion analysis model
            self.emotion_analyzer = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True
            )
            
            # Load multilingual model
            self.multilingual_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                return_all_scores=True
            )
            
            # Mental health keywords and patterns
            self.mental_health_keywords = {
                'depression': [
                    'depressed', 'sad', 'hopeless', 'empty', 'worthless', 'guilty',
                    'tired', 'fatigue', 'sleep', 'insomnia', 'appetite', 'concentration',
                    'death', 'suicide', 'end it all', 'can\'t go on'
                ],
                'anxiety': [
                    'anxious', 'worried', 'nervous', 'panic', 'fear', 'scared',
                    'restless', 'tension', 'heart racing', 'sweating', 'trembling',
                    'overthinking', 'catastrophic', 'worst case'
                ],
                'stress': [
                    'stressed', 'overwhelmed', 'pressure', 'burden', 'exhausted',
                    'burned out', 'can\'t cope', 'too much', 'breaking point'
                ],
                'ptsd': [
                    'flashback', 'nightmare', 'trauma', 'triggered', 'avoidance',
                    'hypervigilant', 'jumpy', 'startled', 'memories', 'intrusive'
                ],
                'bipolar': [
                    'manic', 'mania', 'mood swing', 'high energy', 'euphoric',
                    'grandiose', 'impulsive', 'racing thoughts', 'up and down'
                ]
            }
            
            logger.info("Text analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing text analyzer: {str(e)}")
            raise
    
    def detect_language(self, text: str) -> str:
        """Detect language of input text"""
        try:
            # Simple language detection based on common words
            language_patterns = {
                'en': ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with'],
                'es': ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no'],
                'fr': ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir'],
                'de': ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich'],
                'hi': ['है', 'के', 'में', 'की', 'को', 'से', 'और', 'एक', 'का', 'पर']
            }
            
            text_lower = text.lower()
            scores = {}
            
            for lang, words in language_patterns.items():
                score = sum(1 for word in words if word in text_lower)
                scores[lang] = score
            
            detected_lang = max(scores, key=scores.get) if max(scores.values()) > 0 else 'en'
            return detected_lang
            
        except Exception as e:
            logger.warning(f"Language detection failed: {str(e)}")
            return 'en'
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        return text.strip()
    
    def analyze_sentiment(self, text: str, language: str = 'en') -> Dict:
        """Analyze sentiment of text"""
        try:
            if language == 'en':
                results = self.sentiment_analyzer(text)
            else:
                results = self.multilingual_analyzer(text)
            
            # Convert to standardized format
            sentiment_scores = {}
            for result in results[0]:
                label = result['label'].lower()
                if label in ['positive', 'pos']:
                    sentiment_scores['positive'] = result['score']
                elif label in ['negative', 'neg']:
                    sentiment_scores['negative'] = result['score']
                elif label in ['neutral']:
                    sentiment_scores['neutral'] = result['score']
            
            # Determine dominant sentiment
            dominant_sentiment = max(sentiment_scores, key=sentiment_scores.get)
            
            return {
                'scores': sentiment_scores,
                'dominant': dominant_sentiment,
                'confidence': sentiment_scores[dominant_sentiment]
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            return {
                'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                'dominant': 'neutral',
                'confidence': 0.34
            }
    
    def analyze_emotions(self, text: str) -> Dict:
        """Analyze emotions in text"""
        try:
            results = self.emotion_analyzer(text)
            
            # Extract emotion scores
            emotion_scores = {}
            for result in results[0]:
                emotion_scores[result['label']] = result['score']
            
            # Get top 3 emotions
            top_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            
            return {
                'all_scores': emotion_scores,
                'top_emotions': top_emotions,
                'dominant_emotion': top_emotions[0][0] if top_emotions else 'neutral',
                'dominant_score': top_emotions[0][1] if top_emotions else 0.0
            }
            
        except Exception as e:
            logger.error(f"Emotion analysis failed: {str(e)}")
            return {
                'all_scores': {'neutral': 1.0},
                'top_emotions': [('neutral', 1.0)],
                'dominant_emotion': 'neutral',
                'dominant_score': 1.0
            }
    
    def detect_mental_health_indicators(self, text: str) -> Dict:
        """Detect mental health-related keywords and patterns"""
        text_lower = text.lower()
        indicators = {}
        
        for condition, keywords in self.mental_health_keywords.items():
            matched_keywords = []
            for keyword in keywords:
                if keyword in text_lower:
                    matched_keywords.append(keyword)
            
            if matched_keywords:
                indicators[condition] = {
                    'keywords': matched_keywords,
                    'count': len(matched_keywords),
                    'severity_score': min(len(matched_keywords) / len(keywords), 1.0)
                }
        
        return indicators
    
    def calculate_text_risk_score(self, text: str) -> float:
        """Calculate overall risk score based on text analysis"""
        sentiment = self.analyze_sentiment(text)
        emotions = self.analyze_emotions(text)
        indicators = self.detect_mental_health_indicators(text)
        
        # Base risk from sentiment (negative sentiment increases risk)
        sentiment_risk = sentiment['scores'].get('negative', 0) * 0.3
        
        # Risk from emotions (sadness, anger, fear increase risk)
        high_risk_emotions = ['sadness', 'anger', 'fear', 'disgust']
        emotion_risk = 0
        for emotion, score in emotions['all_scores'].items():
            if emotion in high_risk_emotions:
                emotion_risk += score * 0.2
        
        # Risk from mental health indicators
        indicator_risk = 0
        for condition, data in indicators.items():
            if condition in ['depression', 'anxiety']:
                indicator_risk += data['severity_score'] * 0.3
            else:
                indicator_risk += data['severity_score'] * 0.2
        
        total_risk = min(sentiment_risk + emotion_risk + indicator_risk, 1.0)
        return total_risk
    
    def analyze_text(self, text: str, language: Optional[str] = None) -> Dict:
        """Comprehensive text analysis"""
        if not text or len(text.strip()) == 0:
            return self._empty_analysis()
        
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Detect language if not provided
            if not language:
                language = self.detect_language(processed_text)
            
            # Perform all analyses
            sentiment = self.analyze_sentiment(processed_text, language)
            emotions = self.analyze_emotions(processed_text)
            indicators = self.detect_mental_health_indicators(processed_text)
            risk_score = self.calculate_text_risk_score(processed_text)
            
            return {
                'original_text': text,
                'processed_text': processed_text,
                'language': language,
                'sentiment': sentiment,
                'emotions': emotions,
                'mental_health_indicators': indicators,
                'risk_score': risk_score,
                'word_count': len(processed_text.split()),
                'analysis_timestamp': None
            }
            
        except Exception as e:
            logger.error(f"Text analysis failed: {str(e)}")
            return self._empty_analysis()
    
    def _empty_analysis(self) -> Dict:
        """Return empty analysis structure"""
        return {
            'original_text': '',
            'processed_text': '',
            'language': 'en',
            'sentiment': {'scores': {'neutral': 1.0}, 'dominant': 'neutral', 'confidence': 1.0},
            'emotions': {'all_scores': {'neutral': 1.0}, 'top_emotions': [('neutral', 1.0)], 
                        'dominant_emotion': 'neutral', 'dominant_score': 1.0},
            'mental_health_indicators': {},
            'risk_score': 0.0,
            'word_count': 0,
            'analysis_timestamp': None
        }

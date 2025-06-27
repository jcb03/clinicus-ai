from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class TextAnalyzer:
    def __init__(self):
        """Initialize text analysis models with Hindi support"""
        try:
            # Load sentiment analysis model (English) - FIXED: top_k instead of return_all_scores
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                top_k=None
            )
            
            # Load Hindi sentiment analysis model
            try:
                self.hindi_sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="l3cube-pune/hindi-sentiment-analysis-roberta",
                    top_k=None
                )
            except:
                logger.warning("Hindi sentiment model not available, using multilingual fallback")
                self.hindi_sentiment_analyzer = None
            
            # Load emotion analysis model - FIXED: top_k instead of return_all_scores
            self.emotion_analyzer = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None
            )
            
            # Load multilingual model - FIXED: top_k instead of return_all_scores
            self.multilingual_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                top_k=None
            )
            
            # Enhanced mental health keywords with Hindi support
            self.mental_health_keywords = {
                'depression': {
                    'en': ['depressed', 'sad', 'hopeless', 'empty', 'worthless', 'guilty',
                           'tired', 'fatigue', 'sleep', 'insomnia', 'appetite', 'concentration',
                           'death', 'suicide', 'end it all', 'can\'t go on', 'no energy'],
                    'hi': ['उदास', 'निराश', 'खाली', 'बेकार', 'दोषी', 'थका हुआ',
                           'नींद नहीं', 'भूख नहीं', 'ध्यान नहीं', 'मौत', 'आत्महत्या',
                           'जीना नहीं चाहता', 'कोई उम्मीद नहीं', 'अकेला']
                },
                'anxiety': {
                    'en': ['anxious', 'worried', 'nervous', 'panic', 'fear', 'scared',
                           'restless', 'tension', 'heart racing', 'sweating', 'trembling',
                           'overthinking', 'catastrophic', 'worst case', 'can\'t relax'],
                    'hi': ['चिंतित', 'परेशान', 'घबराया', 'डर', 'बेचैन', 'तनाव',
                           'दिल की धड़कन', 'पसीना', 'कांपना', 'ज्यादा सोचना',
                           'आराम नहीं मिलता', 'घबराहट']
                },
                'stress': {
                    'en': ['stressed', 'overwhelmed', 'pressure', 'burden', 'exhausted',
                           'burned out', 'can\'t cope', 'too much', 'breaking point'],
                    'hi': ['तनावग्रस्त', 'दबाव', 'बोझ', 'थका हुआ', 'परेशान',
                           'बर्दाश्त नहीं हो रहा', 'बहुत ज्यादा', 'हद से ज्यादा']
                },
                'ptsd': {
                    'en': ['flashback', 'nightmare', 'trauma', 'triggered', 'avoidance',
                           'hypervigilant', 'jumpy', 'startled', 'memories', 'intrusive'],
                    'hi': ['बुरी यादें', 'डरावने सपने', 'आघात', 'पुरानी यादें',
                           'बचना', 'चौकन्ना', 'घबराना', 'अचानक डर जाना']
                },
                'bipolar': {
                    'en': ['manic', 'mania', 'mood swing', 'high energy', 'euphoric',
                           'grandiose', 'impulsive', 'racing thoughts', 'up and down'],
                    'hi': ['मूड बदलना', 'उत्साह', 'अचानक खुशी', 'बहुत एनर्जी',
                           'तेज सोचना', 'ऊपर-नीचे', 'अचानक बदलाव']
                }
            }
            
            logger.info("Text analyzer with Hindi support initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing text analyzer: {str(e)}")
            raise
    
    def detect_language(self, text: str) -> str:
        """Enhanced language detection with Hindi support"""
        try:
            # Check for Hindi characters
            hindi_chars = re.findall(r'[\u0900-\u097F]', text)
            if len(hindi_chars) > len(text) * 0.3:  # If 30% or more are Hindi characters
                return 'hi'
            
            # English detection patterns
            english_patterns = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with']
            hindi_patterns = ['है', 'के', 'में', 'की', 'को', 'से', 'और', 'एक', 'का', 'पर']
            
            text_lower = text.lower()
            
            english_score = sum(1 for word in english_patterns if word in text_lower)
            hindi_score = sum(1 for word in hindi_patterns if word in text_lower)
            
            return 'hi' if hindi_score > english_score else 'en'
            
        except Exception as e:
            logger.warning(f"Language detection failed: {str(e)}")
            return 'en'
    
    def analyze_sentiment(self, text: str, language: str = 'en') -> Dict:
        """Analyze sentiment with Hindi support - FIXED: Handle new API format"""
        try:
            if language == 'hi' and self.hindi_sentiment_analyzer:
                results = self.hindi_sentiment_analyzer(text)
            elif language == 'en':
                results = self.sentiment_analyzer(text)
            else:
                results = self.multilingual_analyzer(text)
            
            # FIXED: Handle both old and new format
            sentiment_scores = {}
            
            # Handle list format (when top_k=None returns all scores)
            if isinstance(results, list) and len(results) > 0:
                for result in results:
                    label = result['label'].lower()
                    score = result['score']
                    
                    if label in ['positive', 'pos', 'positive sentiment']:
                        sentiment_scores['positive'] = score
                    elif label in ['negative', 'neg', 'negative sentiment']:
                        sentiment_scores['negative'] = score
                    elif label in ['neutral', 'neutral sentiment']:
                        sentiment_scores['neutral'] = score
            
            # Ensure all sentiment types are present
            if 'positive' not in sentiment_scores:
                sentiment_scores['positive'] = 0.33
            if 'negative' not in sentiment_scores:
                sentiment_scores['negative'] = 0.33
            if 'neutral' not in sentiment_scores:
                sentiment_scores['neutral'] = 0.34
            
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
    
    def detect_mental_health_indicators(self, text: str, language: str = 'en') -> Dict:
        """Enhanced mental health detection with Hindi support"""
        text_lower = text.lower()
        indicators = {}
        
        for condition, lang_keywords in self.mental_health_keywords.items():
            keywords = lang_keywords.get(language, lang_keywords.get('en', []))
            matched_keywords = []
            
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    matched_keywords.append(keyword)
            
            if matched_keywords:
                # Enhanced severity calculation
                base_severity = len(matched_keywords) / len(keywords)
                
                # Weight critical keywords more
                critical_keywords = {
                    'depression': ['suicide', 'आत्महत्या', 'death', 'मौत', 'end it all'],
                    'anxiety': ['panic', 'घबराहट', 'can\'t breathe'],
                    'stress': ['breaking point', 'हद से ज्यादा'],
                    'ptsd': ['trauma', 'आघात', 'flashback'],
                    'bipolar': ['manic', 'racing thoughts']
                }
                
                critical_matches = sum(1 for kw in matched_keywords 
                                     if kw in critical_keywords.get(condition, []))
                
                severity_score = min(base_severity + (critical_matches * 0.2), 1.0)
                
                indicators[condition] = {
                    'keywords': matched_keywords,
                    'count': len(matched_keywords),
                    'severity_score': severity_score,
                    'critical_indicators': critical_matches > 0
                }
        
        return indicators
    
    def calculate_text_risk_score(self, text: str, language: str = 'en') -> float:
        """Enhanced risk calculation with focus on primary concerns"""
        sentiment = self.analyze_sentiment(text, language)
        emotions = self.analyze_emotions(text)
        indicators = self.detect_mental_health_indicators(text, language)
        
        # Base risk from sentiment (negative sentiment increases risk)
        sentiment_risk = sentiment['scores'].get('negative', 0) * 0.4  # Increased weight
        
        # Risk from emotions
        high_risk_emotions = ['sadness', 'anger', 'fear', 'disgust']
        emotion_risk = 0
        for emotion, score in emotions['all_scores'].items():
            if emotion in high_risk_emotions:
                emotion_risk += score * 0.3  # Increased weight
        
        # Enhanced risk from mental health indicators
        indicator_risk = 0
        for condition, data in indicators.items():
            base_risk = data['severity_score'] * 0.3
            
            # Critical indicator bonus
            if data.get('critical_indicators', False):
                base_risk *= 1.5
            
            # Primary concern conditions get higher weight
            if condition in ['depression', 'anxiety', 'ptsd']:
                base_risk *= 1.2
            
            indicator_risk += base_risk
        
        total_risk = min(sentiment_risk + emotion_risk + indicator_risk, 1.0)
        return total_risk
    
    def analyze_emotions(self, text: str) -> Dict:
        """Analyze emotions in text - FIXED: Handle new API format"""
        try:
            results = self.emotion_analyzer(text)
            
            # FIXED: Handle list format properly
            emotion_scores = {}
            if isinstance(results, list) and len(results) > 0:
                for result in results:
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
    
    def analyze_text(self, text: str, language: Optional[str] = None) -> Dict:
        """Comprehensive text analysis with Hindi support"""
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
            indicators = self.detect_mental_health_indicators(processed_text, language)
            risk_score = self.calculate_text_risk_score(processed_text, language)
            
            return {
                'original_text': text,
                'processed_text': processed_text,
                'language': language,
                'sentiment': sentiment,
                'emotions': emotions,
                'mental_health_indicators': indicators,
                'risk_score': risk_score,
                'word_count': len(processed_text.split()),
                'primary_concerns': self._identify_primary_concerns(indicators, sentiment, emotions),
                'analysis_timestamp': None
            }
            
        except Exception as e:
            logger.error(f"Text analysis failed: {str(e)}")
            return self._empty_analysis()
    
    def _identify_primary_concerns(self, indicators: Dict, sentiment: Dict, emotions: Dict) -> List[Dict]:
        """Identify primary mental health concerns with confidence scores"""
        concerns = []
        
        for condition, data in indicators.items():
            concern_score = data['severity_score']
            
            # Boost score based on sentiment and emotions
            if sentiment['dominant'] == 'negative':
                concern_score += sentiment['confidence'] * 0.2
            
            # Boost based on relevant emotions
            emotion_boost = {
                'depression': ['sadness'],
                'anxiety': ['fear'],
                'stress': ['anger'],
                'ptsd': ['fear', 'sadness'],
                'bipolar': ['anger', 'joy']
            }
            
            for emotion in emotion_boost.get(condition, []):
                if emotion in emotions['all_scores']:
                    concern_score += emotions['all_scores'][emotion] * 0.15
            
            concerns.append({
                'condition': condition,
                'severity_score': min(concern_score, 1.0),
                'confidence': min(concern_score * 100, 100),
                'critical': data.get('critical_indicators', False)
            })
        
        # Sort by severity and return top concerns
        concerns.sort(key=lambda x: x['severity_score'], reverse=True)
        return concerns[:3]  # Top 3 concerns
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for both English and Hindi"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        return text.strip()
    
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
            'primary_concerns': [],
            'analysis_timestamp': None
        }

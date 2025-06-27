from transformers import pipeline
import torch
import re
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class TextAnalyzer:
    def __init__(self):
        """Initialize text analysis models with CRITICAL detection"""
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                top_k=None
            )
            
            self.emotion_analyzer = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None
            )
            
            # CRITICAL KEYWORDS - Enhanced for immediate detection
            self.mental_health_keywords = {
                'self_harm': [
                    'kill myself', 'suicide', 'want to die', 'end my life', 'hurt myself', 
                    'cut myself', 'self harm', 'better off dead', 'bullet into my skull',
                    'put a bullet', 'gun to my head', 'hang myself', 'jump off',
                    'overdose', 'slit my wrists', 'end it all', 'fucking die',
                    'want to fucking die', 'going to kill myself', 'plan to kill myself',
                    'shoot myself', 'stab myself', 'poison myself', 'drown myself'
                ],
                'depression': [
                    'depressed', 'sad', 'hopeless', 'empty', 'worthless', 'tired',
                    'very sad', 'extremely sad', 'feel awful', 'hate my life',
                    'life sucks', 'no point', 'meaningless', 'give up', 'can\'t go on'
                ],
                'anxiety': [
                    'anxious', 'worried', 'nervous', 'panic', 'fear', 'scared',
                    'panic attack', 'heart racing', 'can\'t breathe', 'overwhelmed'
                ],
                'stress': [
                    'stressed', 'overwhelmed', 'pressure', 'burned out', 'can\'t cope'
                ]
            }
            
            logger.info("Text analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing text analyzer: {str(e)}")
            raise
    
    def detect_mental_health_indicators(self, text: str) -> Dict:
        """CRITICAL DETECTION - Enhanced for immediate high-risk identification"""
        if not text or len(text.strip()) == 0:
            return {}
        
        text_lower = text.lower().strip()
        indicators = {}
        
        # ULTRA-CRITICAL PATTERNS - Immediate maximum detection
        ultra_critical_patterns = [
            'kill myself', 'suicide', 'want to die', 'end my life', 'better off dead',
            'bullet into my skull', 'put a bullet', 'gun to my head', 'hang myself',
            'jump off', 'overdose', 'slit my wrists', 'fucking die', 'want to fucking die',
            'shoot myself', 'stab myself', 'poison myself', 'drown myself'
        ]
        
        # Check for ultra-critical patterns FIRST
        found_ultra_critical = []
        for pattern in ultra_critical_patterns:
            if pattern in text_lower:
                found_ultra_critical.append(pattern)
        
        # If ANY ultra-critical pattern found, IMMEDIATE maximum severity
        if found_ultra_critical:
            indicators['self_harm'] = {
                'keywords': found_ultra_critical,
                'count': len(found_ultra_critical),
                'severity_score': 0.98,  # MAXIMUM severity
                'critical_indicators': True,
                'confidence': 98.0
            }
            logger.critical(f"🚨 ULTRA-CRITICAL SELF-HARM DETECTED: {found_ultra_critical}")
        
        # Regular keyword detection for other conditions
        for condition, keywords in self.mental_health_keywords.items():
            if condition == 'self_harm' and 'self_harm' in indicators:
                continue  # Skip if already detected with ultra-critical
                
            found_keywords = []
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    found_keywords.append(keyword)
            
            if found_keywords:
                severity_score = min(0.9, len(found_keywords) / max(len(keywords), 3) + 0.5)
                confidence = min(severity_score * 100, 100)
                
                indicators[condition] = {
                    'keywords': found_keywords[:5],
                    'count': len(found_keywords),
                    'severity_score': float(severity_score),
                    'critical_indicators': True,
                    'confidence': float(confidence)
                }
        
        return indicators
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Enhanced sentiment with crisis override"""
        try:
            text_lower = text.lower()
            
            # CRITICAL OVERRIDE for severe cases
            critical_phrases = ['kill myself', 'bullet into my skull', 'want to die', 'suicide', 'end my life']
            if any(phrase in text_lower for phrase in critical_phrases):
                return {
                    'scores': {'positive': 0.02, 'negative': 0.95, 'neutral': 0.03},
                    'dominant': 'negative',
                    'confidence': 0.95
                }
            
            # Regular analysis
            results = self.sentiment_analyzer(text)
            sentiment_scores = {}
            
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict) and 'label' in item and 'score' in item:
                        label = str(item['label']).lower().strip()
                        score = float(item['score'])
                        
                        if 'positive' in label or label == 'label_2':
                            sentiment_scores['positive'] = score
                        elif 'negative' in label or label == 'label_0':
                            sentiment_scores['negative'] = score
                        elif 'neutral' in label or label == 'label_1':
                            sentiment_scores['neutral'] = score
            
            if not sentiment_scores:
                sentiment_scores = {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
            else:
                # Fill missing
                if 'positive' not in sentiment_scores:
                    sentiment_scores['positive'] = 0.2
                if 'negative' not in sentiment_scores:
                    sentiment_scores['negative'] = 0.3
                if 'neutral' not in sentiment_scores:
                    sentiment_scores['neutral'] = 0.5
                
                # Normalize
                total = sum(sentiment_scores.values())
                if total > 0:
                    sentiment_scores = {k: v/total for k, v in sentiment_scores.items()}
            
            dominant = max(sentiment_scores, key=sentiment_scores.get)
            return {
                'scores': sentiment_scores,
                'dominant': dominant,
                'confidence': sentiment_scores[dominant]
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            return {
                'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                'dominant': 'neutral',
                'confidence': 0.34
            }
    
    def analyze_emotions(self, text: str) -> Dict:
        """Enhanced emotions with crisis override"""
        try:
            text_lower = text.lower()
            
            # CRITICAL OVERRIDE
            critical_phrases = ['kill myself', 'bullet into my skull', 'want to die', 'suicide', 'end my life']
            if any(phrase in text_lower for phrase in critical_phrases):
                return {
                    'all_scores': {'sadness': 0.90, 'anger': 0.80, 'fear': 0.70, 'neutral': 0.05},
                    'top_emotions': [('sadness', 0.90), ('anger', 0.80), ('fear', 0.70)],
                    'dominant_emotion': 'sadness',
                    'dominant_score': 0.90
                }
            
            # Regular analysis
            results = self.emotion_analyzer(text)
            emotion_scores = {}
            
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict) and 'label' in item and 'score' in item:
                        emotion_scores[item['label']] = item['score']
            
            if not emotion_scores:
                emotion_scores = {'neutral': 1.0}
            
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
    
    def calculate_text_risk_score(self, text: str) -> float:
        """CRITICAL risk calculation"""
        try:
            text_lower = text.lower()
            
            # IMMEDIATE MAXIMUM RISK
            critical_phrases = ['kill myself', 'bullet into my skull', 'want to die', 'suicide', 'end my life', 'gun to my head']
            if any(phrase in text_lower for phrase in critical_phrases):
                return 0.98  # MAXIMUM risk
            
            # Regular calculation
            sentiment = self.analyze_sentiment(text)
            emotions = self.analyze_emotions(text)
            indicators = self.detect_mental_health_indicators(text)
            
            risk_score = 0.0
            
            # Risk from sentiment
            sentiment_risk = sentiment['scores'].get('negative', 0) * 0.4
            
            # Risk from emotions
            high_risk_emotions = ['sadness', 'anger', 'fear']
            emotion_risk = 0
            for emotion, score in emotions['all_scores'].items():
                if emotion in high_risk_emotions:
                    emotion_risk += score * 0.3
            
            # Risk from indicators
            indicator_risk = 0
            for condition, data in indicators.items():
                base_risk = data['severity_score'] * 0.6
                if condition == 'self_harm':
                    base_risk *= 3.0  # Maximum weight for self-harm
                elif data.get('critical_indicators', False):
                    base_risk *= 2.0
                indicator_risk += base_risk
            
            total_risk = min(sentiment_risk + emotion_risk + indicator_risk, 1.0)
            return max(total_risk, 0.0)
            
        except Exception as e:
            logger.error(f"Risk calculation failed: {str(e)}")
            return 0.0
    
    def _identify_primary_concerns(self, indicators: Dict, sentiment: Dict, emotions: Dict) -> List[Dict]:
        """Primary concerns with self-harm priority"""
        concerns = []
        
        for condition, data in indicators.items():
            concern_score = data['severity_score']
            
            # SELF-HARM gets absolute priority
            if condition == 'self_harm':
                concern_score = max(concern_score, 0.95)  # Ensure high score
            
            # Boost based on sentiment
            if sentiment['dominant'] == 'negative' and concern_score > 0:
                concern_score += sentiment['confidence'] * 0.2
            
            # Boost based on emotions
            emotion_boost = {
                'self_harm': ['sadness', 'anger', 'fear'],
                'depression': ['sadness'],
                'anxiety': ['fear'],
                'stress': ['anger']
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
        
        # Sort by critical status and severity
        concerns.sort(key=lambda x: (x['critical'], x['severity_score']), reverse=True)
        return concerns[:3]
    
    def analyze_text(self, text: str, language: Optional[str] = None) -> Dict:
        """Main analysis with critical detection"""
        if not text or len(text.strip()) == 0:
            return self._empty_analysis()
        
        try:
            processed_text = text.strip()
            
            sentiment = self.analyze_sentiment(processed_text)
            emotions = self.analyze_emotions(processed_text)
            indicators = self.detect_mental_health_indicators(processed_text)
            risk_score = self.calculate_text_risk_score(processed_text)
            primary_concerns = self._identify_primary_concerns(indicators, sentiment, emotions)
            
            return {
                'original_text': text,
                'processed_text': processed_text,
                'language': 'en',
                'sentiment': sentiment,
                'emotions': emotions,
                'mental_health_indicators': indicators,
                'risk_score': risk_score,
                'word_count': len(processed_text.split()),
                'primary_concerns': primary_concerns,
                'analysis_timestamp': None,
                'analysis_successful': True
            }
            
        except Exception as e:
            logger.error(f"Text analysis failed: {str(e)}")
            return self._empty_analysis()
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not text:
            return ""
        
        try:
            # Remove excessive whitespace
            text = ' '.join(text.split())
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            
            # Remove excessive punctuation
            text = re.sub(r'[!]{2,}', '!', text)
            text = re.sub(r'[?]{2,}', '?', text)
            text = re.sub(r'[.]{3,}', '...', text)
            
            return text.strip()
            
        except Exception as e:
            logger.warning(f"Text preprocessing failed: {e}")
            return text.strip()
    
    def _empty_analysis(self) -> Dict:
        """Empty analysis structure"""
        return {
            'original_text': '',
            'processed_text': '',
            'language': 'en',
            'sentiment': {
                'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                'dominant': 'neutral',
                'confidence': 0.34
            },
            'emotions': {
                'all_scores': {'neutral': 1.0},
                'top_emotions': [('neutral', 1.0)],
                'dominant_emotion': 'neutral',
                'dominant_score': 1.0
            },
            'mental_health_indicators': {},
            'risk_score': 0.0,
            'word_count': 0,
            'primary_concerns': [],
            'analysis_timestamp': None,
            'analysis_successful': False
        }

# Test function
if __name__ == "__main__":
    analyzer = TextAnalyzer()
    
    # Test critical case
    test_text = "hi i want to kill myself and put a bullet into my skull"
    result = analyzer.analyze_text(test_text)
    
    print("🚨 CRITICAL TEST RESULTS:")
    print(f"Primary concerns: {result['primary_concerns']}")
    print(f"Risk score: {result['risk_score']}")
    print(f"Dominant emotion: {result['emotions']['dominant_emotion']}")
    print(f"Sentiment: {result['sentiment']['dominant']}")
    
    if result['primary_concerns']:
        primary = result['primary_concerns'][0]
        print(f"✅ Primary concern: {primary['condition']} ({primary['confidence']:.1f}%)")
    else:
        print("❌ No primary concerns detected - CRITICAL ERROR!")

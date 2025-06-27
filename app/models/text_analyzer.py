from transformers import pipeline
import torch
import re
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class TextAnalyzer:
    def __init__(self):
        """Initialize text analysis models - English only with enhanced mental health detection"""
        try:
            # Load sentiment analysis model
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                top_k=None
            )
            
            # Load emotion analysis model
            self.emotion_analyzer = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None
            )
            
            # CRITICAL: Enhanced mental health keywords with immediate detection patterns
            self.mental_health_keywords = {
                'depression': [
                    'depressed', 'sad', 'hopeless', 'empty', 'worthless', 'tired', 'fatigue', 
                    'sleep problems', 'insomnia', 'death', 'suicide', 'end it all', 
                    'can\'t go on', 'no energy', 'exhausted', 'meaningless', 'pointless', 
                    'give up', 'hate myself', 'burden', 'better off dead', 'no hope',
                    'nothing matters', 'feel like dying', 'want to disappear', 
                    'life is not worth living', 'down', 'low mood', 'very sad',
                    'extremely sad', 'so sad', 'really sad', 'super sad', 'feel awful',
                    'feel terrible', 'hate my life', 'life is meaningless', 'life sucks'
                ],
                'self_harm': [
                    'kill myself', 'suicide', 'want to die', 'end my life', 'hurt myself', 
                    'cut myself', 'self harm', 'cutting', 'razor', 'burn myself', 
                    'punish myself', 'deserve pain', 'self injury', 'harm myself',
                    'cut my arms', 'cut my wrists', 'self mutilation', 'want to cut',
                    'going to kill myself', 'plan to kill myself', 'thinking of killing myself',
                    # CRITICAL PHRASES FOR IMMEDIATE DETECTION
                    'jump off', 'jump from', 'break my skull', 'break my head',
                    'smash my head', 'fucking kill myself', 'fucking die', 'want to fucking die',
                    'go on top', 'jump and break', 'fucking break my skull'
                ],
                'anxiety': [
                    'anxious', 'worried', 'nervous', 'panic', 'fear', 'scared',
                    'restless', 'tension', 'heart racing', 'overthinking', 'catastrophic',
                    'worst case', 'can\'t relax', 'on edge', 'jittery', 'panic attack',
                    'heart pounding', 'shortness of breath', 'dizzy', 'sweating',
                    'trembling', 'shaking', 'paranoid', 'hypervigilant', 'constant worry'
                ],
                'stress': [
                    'stressed', 'overwhelmed', 'pressure', 'exhausted', 
                    'burned out', 'can\'t cope', 'too much', 'breaking point',
                    'overloaded', 'swamped', 'drowning', 'suffocating', 
                    'cracking under pressure', 'at my limit', 'falling apart',
                    'can\'t handle', 'under pressure', 'breaking down'
                ],
                'ptsd': [
                    'flashback', 'nightmare', 'trauma', 'triggered', 'avoidance',
                    'hypervigilant', 'memories', 'intrusive', 'haunted', 'reliving',
                    'vivid memories', 'can\'t forget', 'happened again', 'jumpy',
                    'startled', 'on guard', 'traumatic', 'disturbing memories'
                ],
                'bipolar': [
                    'manic', 'mania', 'mood swing', 'high energy', 'euphoric',
                    'racing thoughts', 'up and down', 'extreme', 'rollercoaster',
                    'mood changes', 'emotional swings', 'highs and lows'
                ]
            }
            
            logger.info("Text analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing text analyzer: {str(e)}")
            raise
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment with manual override for critical cases"""
        try:
            text_lower = text.lower()
            
            # CRITICAL: Manual override for severe cases
            critical_negative_phrases = [
                'kill myself', 'want to die', 'suicide', 'end my life',
                'break my skull', 'jump off', 'better off dead', 'fucking die'
            ]
            
            if any(phrase in text_lower for phrase in critical_negative_phrases):
                return {
                    'scores': {'positive': 0.05, 'negative': 0.90, 'neutral': 0.05},
                    'dominant': 'negative',
                    'confidence': 0.90
                }
            
            # Strong sadness override
            strong_sadness_phrases = ['very sad', 'extremely sad', 'so sad', 'really sad', 'feel awful', 'feel terrible']
            if any(phrase in text_lower for phrase in strong_sadness_phrases):
                return {
                    'scores': {'positive': 0.10, 'negative': 0.80, 'neutral': 0.10},
                    'dominant': 'negative',
                    'confidence': 0.80
                }
            
            # Original model analysis
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
            
            # Ensure all sentiments exist
            if not sentiment_scores:
                sentiment_scores = {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
            else:
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
        """FIXED emotion analysis with manual override for critical cases"""
        try:
            text_lower = text.lower()
            
            # CRITICAL: Manual detection for severe cases
            if any(phrase in text_lower for phrase in ['kill myself', 'want to die', 'suicide', 'jump off', 'break my skull', 'fucking die']):
                return {
                    'all_scores': {'sadness': 0.85, 'anger': 0.70, 'fear': 0.60, 'neutral': 0.05},
                    'top_emotions': [('sadness', 0.85), ('anger', 0.70), ('fear', 0.60)],
                    'dominant_emotion': 'sadness',
                    'dominant_score': 0.85
                }
            
            # Strong sadness indicators
            if any(phrase in text_lower for phrase in ['very sad', 'extremely sad', 'so sad', 'really sad', 'feel awful', 'feel terrible', 'hate my life']):
                return {
                    'all_scores': {'sadness': 0.90, 'neutral': 0.10},
                    'top_emotions': [('sadness', 0.90), ('neutral', 0.10)],
                    'dominant_emotion': 'sadness',
                    'dominant_score': 0.90
                }
            
            # Anxiety indicators
            if any(phrase in text_lower for phrase in ['panic attack', 'heart racing', 'can\'t breathe', 'very anxious', 'panic']):
                return {
                    'all_scores': {'fear': 0.80, 'sadness': 0.20},
                    'top_emotions': [('fear', 0.80), ('sadness', 0.20)],
                    'dominant_emotion': 'fear',
                    'dominant_score': 0.80
                }
            
            # Original model analysis
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
    
    def detect_mental_health_indicators(self, text: str) -> Dict:
        """CRITICAL FIX: Enhanced detection for severe cases with immediate high-risk identification"""
        if not text or len(text.strip()) == 0:
            return {}
        
        text_lower = text.lower().strip()
        indicators = {}
        
        # CRITICAL: Ultra-high priority pattern matching first
        ultra_critical_patterns = [
            'kill myself', 'suicide', 'want to die', 'end my life', 'better off dead',
            'jump off', 'break my skull', 'fucking die', 'want to fucking die',
            'go on top', 'jump and break', 'fucking break my skull', 'smash my head'
        ]
        
        # Check for ultra-critical patterns first
        found_ultra_critical = []
        for pattern in ultra_critical_patterns:
            if pattern in text_lower:
                found_ultra_critical.append(pattern)
        
        # If ultra-critical patterns found, immediately flag as self-harm
        if found_ultra_critical:
            indicators['self_harm'] = {
                'keywords': found_ultra_critical,
                'count': len(found_ultra_critical),
                'severity_score': 0.95,  # Maximum severity
                'critical_indicators': True,
                'confidence': 95.0
            }
        
        # Regular keyword analysis for all conditions
        for condition, keywords in self.mental_health_keywords.items():
            found_keywords = []
            
            # Direct keyword matching
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    found_keywords.append(keyword)
            
            # Enhanced phrase matching
            if condition == 'depression':
                depression_phrases = [
                    'feel sad', 'very sad', 'extremely sad', 'so sad', 'really sad',
                    'feel empty', 'feel hopeless', 'feel worthless', 'want to die',
                    'life is meaningless', 'no point in living', 'feel depressed',
                    'hate my life', 'nothing matters', 'feel down', 'low mood',
                    'feel awful', 'feel terrible'
                ]
                for phrase in depression_phrases:
                    if phrase in text_lower:
                        found_keywords.append(phrase)
            
            elif condition == 'anxiety':
                anxiety_phrases = [
                    'feel anxious', 'very anxious', 'so worried', 'panic attacks',
                    'anxiety attacks', 'feel nervous', 'heart racing', 'can\'t breathe',
                    'feel scared', 'constant worry', 'overthinking', 'panic disorder'
                ]
                for phrase in anxiety_phrases:
                    if phrase in text_lower:
                        found_keywords.append(phrase)
            
            elif condition == 'stress':
                stress_phrases = [
                    'feel stressed', 'under pressure', 'too much stress',
                    'can\'t cope', 'overwhelmed with', 'breaking down',
                    'burned out', 'at my limit', 'falling apart'
                ]
                for phrase in stress_phrases:
                    if phrase in text_lower:
                        found_keywords.append(phrase)
            
            # Calculate severity if keywords found
            if found_keywords:
                # Base severity calculation
                base_severity = min(len(found_keywords) / max(len(keywords), 3), 1.0)
                
                # Critical keywords identification
                critical_keywords = {
                    'depression': ['suicide', 'kill myself', 'want to die', 'end it all', 'better off dead'],
                    'anxiety': ['panic attack', 'can\'t breathe', 'heart racing'],
                    'stress': ['breaking point', 'can\'t cope', 'falling apart'],
                    'self_harm': ['cut myself', 'hurt myself', 'self harm', 'kill myself'],
                    'ptsd': ['trauma', 'flashback', 'triggered']
                }
                
                critical_matches = sum(1 for kw in found_keywords 
                                     if kw in critical_keywords.get(condition, []))
                
                # Enhanced severity calculation
                if critical_matches > 0:
                    severity_score = max(0.85, base_severity + (critical_matches * 0.4))
                else:
                    severity_score = max(0.45, base_severity + 0.25)
                
                severity_score = min(severity_score, 1.0)
                confidence = min(severity_score * 120, 100)
                
                # Don't override self_harm if already detected with ultra-critical
                if condition == 'self_harm' and 'self_harm' in indicators:
                    continue
                
                indicators[condition] = {
                    'keywords': found_keywords[:5],
                    'count': len(found_keywords),
                    'severity_score': float(severity_score),
                    'critical_indicators': critical_matches > 0,
                    'confidence': float(confidence)
                }
        
        return indicators
    
    def calculate_text_risk_score(self, text: str) -> float:
        """CRITICAL FIX: Enhanced risk calculation with immediate high-risk detection"""
        try:
            text_lower = text.lower()
            
            # IMMEDIATE MAXIMUM RISK for critical phrases
            ultra_critical_phrases = [
                'kill myself', 'suicide', 'want to die', 'end my life',
                'jump off', 'break my skull', 'better off dead', 'fucking die',
                'want to fucking die', 'go on top', 'jump and break'
            ]
            
            if any(phrase in text_lower for phrase in ultra_critical_phrases):
                return 0.95  # Maximum risk
            
            # High risk for severe depression
            severe_depression_phrases = [
                'very sad', 'extremely sad', 'hate my life', 'life is meaningless',
                'no point in living', 'feel awful', 'feel terrible'
            ]
            
            if any(phrase in text_lower for phrase in severe_depression_phrases):
                return 0.75  # High risk
            
            # Regular analysis
            sentiment = self.analyze_sentiment(text)
            emotions = self.analyze_emotions(text)
            indicators = self.detect_mental_health_indicators(text)
            
            risk_score = 0.0
            
            # Risk from negative sentiment
            sentiment_risk = sentiment['scores'].get('negative', 0) * 0.5
            
            # Risk from emotions
            high_risk_emotions = ['sadness', 'anger', 'fear']
            emotion_risk = 0
            for emotion, score in emotions['all_scores'].items():
                if emotion in high_risk_emotions:
                    emotion_risk += score * 0.4
            
            # Risk from mental health indicators
            indicator_risk = 0
            for condition, data in indicators.items():
                base_risk = data['severity_score'] * 0.6
                
                # CRITICAL: Self-harm gets maximum weight
                if condition == 'self_harm':
                    base_risk *= 3.0  # Triple weight for self-harm
                elif data.get('critical_indicators', False):
                    base_risk *= 2.0
                
                indicator_risk += base_risk
            
            total_risk = min(sentiment_risk + emotion_risk + indicator_risk, 1.0)
            return max(total_risk, 0.0)
            
        except Exception as e:
            logger.error(f"Risk calculation failed: {str(e)}")
            return 0.0
    
    def _identify_primary_concerns(self, indicators: Dict, sentiment: Dict, emotions: Dict) -> List[Dict]:
        """Identify primary concerns with critical priority"""
        concerns = []
        
        for condition, data in indicators.items():
            concern_score = data['severity_score']
            
            # CRITICAL: Self-harm always gets highest priority
            if condition == 'self_harm':
                concern_score *= 2.0  # Double the score for self-harm
            
            # Boost based on sentiment
            if sentiment['dominant'] == 'negative' and concern_score > 0:
                concern_score += sentiment['confidence'] * 0.3
            
            # Boost based on emotions
            emotion_boost = {
                'depression': ['sadness'],
                'anxiety': ['fear'],
                'stress': ['anger'],
                'ptsd': ['fear', 'sadness'],
                'self_harm': ['sadness', 'anger', 'fear']
            }
            
            for emotion in emotion_boost.get(condition, []):
                if emotion in emotions['all_scores']:
                    concern_score += emotions['all_scores'][emotion] * 0.2
            
            concerns.append({
                'condition': condition,
                'severity_score': min(concern_score, 1.0),
                'confidence': min(concern_score * 100, 100),
                'critical': data.get('critical_indicators', False)
            })
        
        # Sort by severity score and critical status
        concerns.sort(key=lambda x: (x['critical'], x['severity_score']), reverse=True)
        return concerns[:3]
    
    def analyze_text(self, text: str, language: Optional[str] = None) -> Dict:
        """Main text analysis function with critical detection"""
        if not text or len(text.strip()) == 0:
            return self._empty_analysis()
        
        try:
            processed_text = text.strip()
            
            # Perform analyses
            sentiment = self.analyze_sentiment(processed_text)
            emotions = self.analyze_emotions(processed_text)
            indicators = self.detect_mental_health_indicators(processed_text)
            risk_score = self.calculate_text_risk_score(processed_text)
            
            # Identify primary concerns
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
    
    def _empty_analysis(self) -> Dict:
        """Empty analysis structure"""
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
            'analysis_timestamp': None,
            'analysis_successful': False
        }

# Test function
if __name__ == "__main__":
    analyzer = TextAnalyzer()
    
    # Test critical case
    test_text = "i am feeling very sad, i want to kill myself. i want to go on top of a hill and jump and fucking break my skull"
    result = analyzer.analyze_text(test_text)
    
    print("Test Results:")
    print(f"Primary concerns: {result['primary_concerns']}")
    print(f"Risk score: {result['risk_score']}")
    print(f"Dominant emotion: {result['emotions']['dominant_emotion']}")
    print(f"Sentiment: {result['sentiment']['dominant']}")

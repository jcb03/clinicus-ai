from transformers import pipeline
import torch
import re
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class TextAnalyzer:
    def __init__(self):
        """Initialize text analysis models - English only"""
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
            
            # Enhanced mental health keywords - FIXED AND EXPANDED
            self.mental_health_keywords = {
                'depression': [
                    'depressed', 'sad', 'hopeless', 'empty', 'worthless', 'guilty',
                    'tired', 'fatigue', 'sleeping too much', 'insomnia', 'no appetite', 
                    'concentration problems', 'death', 'suicide', 'kill myself', 'end it all', 
                    'can\'t go on', 'no energy', 'meaningless', 'pointless', 'give up',
                    'hate myself', 'burden', 'better off dead', 'no hope', 'nothing matters',
                    'feel like dying', 'want to disappear', 'life is not worth living'
                ],
                'anxiety': [
                    'anxious', 'worried', 'nervous', 'panic', 'fear', 'scared',
                    'restless', 'tension', 'heart racing', 'sweating', 'trembling',
                    'overthinking', 'catastrophic', 'worst case', 'can\'t relax', 'jittery',
                    'on edge', 'paranoid', 'hypervigilant', 'constant worry', 'dread',
                    'panic attack', 'heart pounding', 'shortness of breath', 'dizzy'
                ],
                'stress': [
                    'stressed', 'overwhelmed', 'pressure', 'burden', 'exhausted',
                    'burned out', 'can\'t cope', 'too much', 'breaking point', 'swamped',
                    'overloaded', 'drowning', 'suffocating', 'cracking under pressure',
                    'at my limit', 'falling apart', 'can\'t handle'
                ],
                'ptsd': [
                    'flashback', 'nightmare', 'trauma', 'triggered', 'avoidance',
                    'hypervigilant', 'jumpy', 'startled', 'memories', 'intrusive', 'haunted',
                    'reliving', 'vivid memories', 'can\'t forget', 'happened again'
                ],
                'self_harm': [
                    'cut myself', 'hurt myself', 'self harm', 'cutting', 'razor',
                    'burn myself', 'punish myself', 'deserve pain', 'self injury'
                ]
            }
            
            logger.info("Text analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing text analyzer: {str(e)}")
            raise
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment - FIXED"""
        try:
            results = self.sentiment_analyzer(text)
            sentiment_scores = {}
            
            # Process results properly
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict) and 'label' in item and 'score' in item:
                        label = str(item['label']).lower().strip()
                        score = float(item['score'])
                        
                        # Map labels correctly
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
                # Fill missing sentiments
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
        """Analyze emotions - FIXED"""
        try:
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
        """ENHANCED mental health detection - FIXED"""
        if not text or len(text.strip()) == 0:
            return {}
        
        text_lower = text.lower().strip()
        indicators = {}
        
        # More aggressive keyword matching
        for condition, keywords in self.mental_health_keywords.items():
            matched_keywords = []
            
            # Check for keyword matches
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    matched_keywords.append(keyword)
            
            # Also check for partial matches and variations
            if condition == 'depression':
                depression_phrases = ['feel sad', 'very sad', 'extremely sad', 'so sad', 
                                    'feel empty', 'feel hopeless', 'feel worthless', 
                                    'want to die', 'life is meaningless', 'no point in living']
                for phrase in depression_phrases:
                    if phrase in text_lower:
                        matched_keywords.append(phrase)
            
            elif condition == 'anxiety':
                anxiety_phrases = ['feel anxious', 'very anxious', 'so worried', 'panic attacks',
                                 'anxiety attacks', 'feel nervous', 'heart racing', 'can\'t breathe']
                for phrase in anxiety_phrases:
                    if phrase in text_lower:
                        matched_keywords.append(phrase)
            
            elif condition == 'stress':
                stress_phrases = ['feel stressed', 'under pressure', 'too much stress',
                                'can\'t cope', 'overwhelmed with', 'breaking down']
                for phrase in stress_phrases:
                    if phrase in text_lower:
                        matched_keywords.append(phrase)
            
            # If matches found, calculate severity
            if matched_keywords:
                # More sensitive scoring
                base_severity = min(len(matched_keywords) / max(len(keywords), 5), 1.0)
                
                # Critical keywords boost
                critical_keywords = {
                    'depression': ['suicide', 'kill myself', 'want to die', 'end it all', 'better off dead'],
                    'anxiety': ['panic attack', 'can\'t breathe', 'heart racing'],
                    'stress': ['breaking point', 'can\'t cope', 'falling apart'],
                    'self_harm': ['cut myself', 'hurt myself', 'self harm']
                }
                
                critical_matches = sum(1 for kw in matched_keywords 
                                     if kw in critical_keywords.get(condition, []))
                
                severity_score = min(base_severity + (critical_matches * 0.5), 1.0)
                
                # Ensure minimum severity for any detection
                severity_score = max(severity_score, 0.3)
                
                indicators[condition] = {
                    'keywords': matched_keywords[:5],
                    'count': len(matched_keywords),
                    'severity_score': severity_score,
                    'critical_indicators': critical_matches > 0,
                    'confidence': min(severity_score * 100, 100)
                }
        
        return indicators
    
    def calculate_text_risk_score(self, text: str) -> float:
        """Calculate risk score - FIXED"""
        try:
            sentiment = self.analyze_sentiment(text)
            emotions = self.analyze_emotions(text)
            indicators = self.detect_mental_health_indicators(text)
            
            # Base risk from negative sentiment
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
                
                # Critical indicators
                if data.get('critical_indicators', False):
                    base_risk *= 2.0
                
                indicator_risk += base_risk
            
            total_risk = min(sentiment_risk + emotion_risk + indicator_risk, 1.0)
            return max(total_risk, 0.0)
            
        except Exception as e:
            logger.error(f"Risk calculation failed: {str(e)}")
            return 0.0
    
    def analyze_text(self, text: str) -> Dict:
        """Main text analysis function - FIXED"""
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
    
    def _identify_primary_concerns(self, indicators: Dict, sentiment: Dict, emotions: Dict) -> List[Dict]:
        """Identify primary concerns - FIXED"""
        concerns = []
        
        for condition, data in indicators.items():
            concern_score = data['severity_score']
            
            # Boost based on sentiment
            if sentiment['dominant'] == 'negative' and concern_score > 0:
                concern_score += sentiment['confidence'] * 0.3
            
            # Boost based on emotions
            emotion_boost = {
                'depression': ['sadness'],
                'anxiety': ['fear'],
                'stress': ['anger'],
                'ptsd': ['fear', 'sadness']
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
        
        concerns.sort(key=lambda x: x['severity_score'], reverse=True)
        return concerns[:3]
    
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

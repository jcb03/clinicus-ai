import logging
from typing import Dict, List, Optional, Any
import time 
from datetime import datetime
import numpy as np
import openai
from openai import OpenAI

from models.text_analyzer import TextAnalyzer
from models.audio_analyzer import AudioAnalyzer
from models.video_analyzer import VideoAnalyzer

logger = logging.getLogger(__name__)

class DiagnosisEngine:
    def __init__(self, openai_api_key: str):
        """Initialize the diagnosis engine with all analyzers"""
        try:
            # Initialize OpenAI client properly
            self.openai_api_key = openai_api_key
            self.openai_client = OpenAI(api_key=openai_api_key)
            
            # Initialize analyzers
            self.text_analyzer = TextAnalyzer()
            self.audio_analyzer = AudioAnalyzer()
            self.video_analyzer = VideoAnalyzer()
            
            # FIXED: Enhanced condition definitions INCLUDING self_harm
            self.condition_definitions = {
                'self_harm': {
                    'description': 'Thoughts or actions of self-injury or suicide',
                    'severity_thresholds': {'mild': 0.3, 'moderate': 0.6, 'severe': 0.8},
                    'keywords': ['kill myself', 'suicide', 'self harm', 'hurt myself']
                },
                'depression': {
                    'description': 'Persistent feelings of sadness, hopelessness, and lack of interest in activities',
                    'severity_thresholds': {'mild': 0.3, 'moderate': 0.5, 'severe': 0.7},
                    'keywords': ['sad', 'hopeless', 'worthless', 'tired', 'sleep problems']
                },
                'anxiety': {
                    'description': 'Excessive worry, fear, and physical symptoms like rapid heartbeat',
                    'severity_thresholds': {'mild': 0.25, 'moderate': 0.45, 'severe': 0.65},
                    'keywords': ['worried', 'nervous', 'panic', 'restless', 'tense']
                },
                'stress': {
                    'description': 'Overwhelming pressure and difficulty coping with demands',
                    'severity_thresholds': {'mild': 0.2, 'moderate': 0.4, 'severe': 0.6},
                    'keywords': ['overwhelmed', 'pressure', 'burden', 'exhausted']
                },
                'ptsd': {
                    'description': 'Trauma-related symptoms including flashbacks and avoidance behaviors',
                    'severity_thresholds': {'mild': 0.3, 'moderate': 0.5, 'severe': 0.7},
                    'keywords': ['trauma', 'flashback', 'nightmare', 'triggered']
                },
                'bipolar': {
                    'description': 'Mood swings between depression and mania or hypomania',
                    'severity_thresholds': {'mild': 0.25, 'moderate': 0.45, 'severe': 0.65},
                    'keywords': ['mood swing', 'manic', 'euphoric', 'racing thoughts']
                }
            }
            
            logger.info("Diagnosis engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing diagnosis engine: {str(e)}")
            raise
    
    def analyze_all_inputs(self, text: Optional[str] = None, 
                          audio_file: Optional[str] = None, 
                          video_frame: Optional[np.ndarray] = None) -> Dict:
        """Analyze all provided inputs"""
        results = {}
        
        try:
            # Text analysis
            if text and len(text.strip()) > 0:
                logger.info("Starting text analysis")
                results['text_analysis'] = self.text_analyzer.analyze_text(text)
                logger.info("Text analysis completed")
            
            # Audio analysis
            if audio_file:
                logger.info("Starting audio analysis")
                results['audio_analysis'] = self.audio_analyzer.analyze_audio(audio_file)
                logger.info("Audio analysis completed")
            
            # Video analysis
            if video_frame is not None:
                logger.info("Starting video analysis")
                results['video_analysis'] = self.video_analyzer.analyze_video_frame(video_frame)
                logger.info("Video analysis completed")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in analyze_all_inputs: {str(e)}")
            return {}
    
    def combine_emotion_scores(self, analysis_results: Dict) -> Dict:
        """Combine emotion scores from all modalities"""
        try:
            combined_emotions = {}
            total_weight = 0
            
            # Text emotions (weight: 0.4)
            if 'text_analysis' in analysis_results:
                text_emotions = analysis_results['text_analysis'].get('emotions', {}).get('all_scores', {})
                for emotion, score in text_emotions.items():
                    combined_emotions[emotion] = combined_emotions.get(emotion, 0) + score * 0.4
                total_weight += 0.4
            
            # Audio emotions (weight: 0.35)
            if 'audio_analysis' in analysis_results:
                audio_emotions = analysis_results['audio_analysis'].get('emotions', {}).get('emotions', {})
                for emotion, score in audio_emotions.items():
                    combined_emotions[emotion] = combined_emotions.get(emotion, 0) + score * 0.35
                total_weight += 0.35
            
            # Video emotions (weight: 0.25)
            if 'video_analysis' in analysis_results:
                video_emotions = analysis_results['video_analysis'].get('emotions', {})
                for emotion, score in video_emotions.items():
                    combined_emotions[emotion] = combined_emotions.get(emotion, 0) + score * 0.25
                total_weight += 0.25
            
            # Normalize scores
            if total_weight > 0:
                for emotion in combined_emotions:
                    combined_emotions[emotion] /= total_weight
            
            # Find dominant emotion
            if combined_emotions:
                dominant_emotion = max(combined_emotions, key=combined_emotions.get)
                confidence = combined_emotions[dominant_emotion]
            else:
                dominant_emotion = 'neutral'
                confidence = 1.0
                combined_emotions = {'neutral': 1.0}
            
            return {
                'combined_emotions': combined_emotions,
                'dominant_emotion': dominant_emotion,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error combining emotion scores: {str(e)}")
            return {
                'combined_emotions': {'neutral': 1.0},
                'dominant_emotion': 'neutral',
                'confidence': 1.0
            }
    
    def calculate_condition_scores(self, analysis_results: Dict) -> Dict:
        """FIXED: Calculate scores for ALL mental health conditions including self_harm"""
        try:
            condition_scores = {}
            
            # Initialize ALL conditions with 0 score (including self_harm)
            for condition in self.condition_definitions:
                condition_scores[condition] = 0.0
            
            # CRITICAL: Text-based scoring with DIRECT reading from text analyzer
            if 'text_analysis' in analysis_results:
                text_data = analysis_results['text_analysis']
                
                # FIXED: Read ALL mental health indicators from text analyzer
                indicators = text_data.get('mental_health_indicators', {})
                logger.info(f"Text indicators found: {list(indicators.keys())}")
                
                for condition, data in indicators.items():
                    # CRITICAL: Include self_harm and all other conditions
                    if condition in self.condition_definitions:
                        severity_score = data.get('severity_score', 0)
                        confidence = data.get('confidence', 0) / 100  # Convert to 0-1
                        
                        # ENHANCED scoring for critical conditions
                        if condition == 'self_harm':
                            condition_scores[condition] = max(severity_score, 0.9)  # Minimum 90% for self-harm
                            logger.critical(f"🚨 SELF-HARM DETECTED: Score = {condition_scores[condition]}")
                        else:
                            condition_scores[condition] = severity_score
                        
                        logger.info(f"Condition '{condition}': severity={severity_score}, confidence={confidence}")
                    else:
                        # Add unknown conditions to our tracking
                        condition_scores[condition] = data.get('severity_score', 0)
                        logger.warning(f"Unknown condition detected: {condition}")
                
                # Sentiment contribution (only for depression/anxiety if no self-harm)
                if condition_scores.get('self_harm', 0) < 0.3:  # Only if no self-harm detected
                    sentiment = text_data.get('sentiment', {})
                    if sentiment.get('dominant') == 'negative':
                        condition_scores['depression'] += sentiment.get('confidence', 0) * 0.2
                        condition_scores['anxiety'] += sentiment.get('confidence', 0) * 0.15
            
            # Audio-based scoring (weight: 0.3)
            if 'audio_analysis' in analysis_results:
                audio_data = analysis_results['audio_analysis']
                mental_health = audio_data.get('mental_health_analysis', {})
                
                detected_conditions = mental_health.get('detected_conditions', {})
                for condition, data in detected_conditions.items():
                    if condition in condition_scores:
                        condition_scores[condition] += data.get('severity', 0) * 0.3
            
            # Video-based scoring (weight: 0.2)
            if 'video_analysis' in analysis_results:
                video_data = analysis_results['video_analysis']
                dominant_emotion = video_data.get('dominant_emotion', '').lower()
                
                emotion_condition_mapping = {
                    'sad': ['depression'],
                    'angry': ['stress'],
                    'fear': ['anxiety', 'ptsd'],
                    'surprise': ['anxiety']
                }
                
                for emotion, conditions in emotion_condition_mapping.items():
                    if emotion in dominant_emotion:
                        for condition in conditions:
                            if condition in condition_scores:
                                condition_scores[condition] += 0.2
            
            # Normalize scores to 0-1 range
            for condition in condition_scores:
                condition_scores[condition] = min(condition_scores[condition], 1.0)
            
            logger.info(f"Final condition scores: {condition_scores}")
            return condition_scores
            
        except Exception as e:
            logger.error(f"Error calculating condition scores: {str(e)}")
            return {condition: 0.0 for condition in self.condition_definitions}
    
    def generate_diagnosis(self, analysis_results: Dict) -> Dict:
        """FIXED: Generate comprehensive diagnosis including self_harm detection"""
        try:
            # Calculate condition scores
            condition_scores = self.calculate_condition_scores(analysis_results)
            
            # Combine emotion analysis
            emotion_analysis = self.combine_emotion_scores(analysis_results)
            
            # FIXED: Get top conditions with LOWER threshold for self_harm
            top_conditions = []
            for condition, score in condition_scores.items():
                # CRITICAL: Different thresholds for different conditions
                min_threshold = 0.05 if condition == 'self_harm' else 0.1
                
                if score > min_threshold:
                    severity = self._determine_severity(condition, score)
                    top_conditions.append({
                        'condition': condition,
                        'score': score,
                        'confidence_percentage': int(score * 100),
                        'severity': severity,
                        'description': self.condition_definitions.get(condition, {}).get('description', 'Mental health concern')
                    })
            
            # CRITICAL: Sort by score with self_harm priority
            top_conditions.sort(key=lambda x: (x['condition'] == 'self_harm', x['score']), reverse=True)
            
            # Generate AI analysis
            ai_analysis = self._generate_ai_diagnosis(analysis_results, top_conditions, emotion_analysis)
            
            # FIXED: Determine overall risk level with self_harm priority
            overall_risk = self._determine_overall_risk(condition_scores)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(top_conditions)
            
            return {
                'top_conditions': top_conditions[:5],
                'emotion_analysis': emotion_analysis,
                'ai_analysis': ai_analysis,
                'overall_risk_level': overall_risk,
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating diagnosis: {str(e)}")
            return self._default_diagnosis()
    
    def _generate_ai_diagnosis(self, analysis_results: Dict, top_conditions: List, emotion_analysis: Dict) -> str:
        """Generate AI-powered diagnosis explanation"""
        try:
            # Prepare context for AI
            context = self._prepare_ai_context(analysis_results, top_conditions, emotion_analysis)
            
            # CRITICAL: Enhanced prompt for self-harm detection
            crisis_detected = any(cond['condition'] == 'self_harm' for cond in top_conditions)
            
            if crisis_detected:
                prompt = f"""CRISIS SITUATION DETECTED - Based on the following mental health analysis, provide an IMMEDIATE crisis response:

{context}

This person has expressed thoughts of self-harm or suicide. Provide:
1. Immediate concern and validation
2. Strong encouragement to seek emergency help
3. Crisis resources and hotlines
4. Supportive but urgent guidance

Keep response under 150 words and prioritize safety."""
            else:
                prompt = f"""Based on the following mental health analysis data, provide a compassionate, professional assessment:

{context}

Please provide:
1. A brief, empathetic summary of the person's current state
2. Key observations from the analysis
3. Supportive guidance and encouragement
4. Clear recommendation to consult with a mental health professional

Keep the response warm, non-judgmental, and under 200 words. Avoid medical diagnoses."""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a compassionate mental health assessment AI that provides supportive, professional insights while always encouraging professional consultation. For crisis situations, prioritize immediate safety."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"AI diagnosis generation failed: {str(e)}")
            return "Unable to generate detailed analysis at this time. Please consult with a mental health professional for comprehensive assessment."
    
    def _prepare_ai_context(self, analysis_results: Dict, top_conditions: List, emotion_analysis: Dict) -> str:
        """Prepare context for AI analysis"""
        context_parts = []
        
        # Text analysis context
        if analysis_results.get('text_analysis'):
            text_data = analysis_results['text_analysis']
            context_parts.append(f"Text Analysis: Sentiment is {text_data.get('sentiment', {}).get('dominant', 'unknown')}, dominant emotion is {text_data.get('emotions', {}).get('dominant_emotion', 'unknown')}")
            
            if text_data.get('mental_health_indicators'):
                indicators = list(text_data['mental_health_indicators'].keys())
                context_parts.append(f"Mental health indicators found: {', '.join(indicators)}")
        
        # Audio analysis context
        if analysis_results.get('audio_analysis'):
            audio_data = analysis_results['audio_analysis']
            context_parts.append(f"Audio Analysis: Voice emotion detected as {audio_data.get('emotions', {}).get('dominant_emotion', 'unknown')}")
            
            if audio_data.get('transcription'):
                transcription = audio_data['transcription'].get('transcription', '')
                if transcription:
                    context_parts.append(f"Speech content: {transcription[:200]}...")
        
        # Video analysis context
        if analysis_results.get('video_analysis'):
            video_data = analysis_results['video_analysis']
            context_parts.append(f"Facial Analysis: Facial expression shows {video_data.get('dominant_emotion', 'unknown')}")
        
        # Top conditions
        if top_conditions:
            conditions_text = ", ".join([f"{cond['condition']} ({cond['score']:.2f})" for cond in top_conditions])
            context_parts.append(f"Top identified concerns: {conditions_text}")
        
        # Dominant emotion
        dominant = emotion_analysis.get('dominant_emotion', 'unknown')
        confidence = emotion_analysis.get('confidence', 0)
        context_parts.append(f"Overall dominant emotion: {dominant} (confidence: {confidence:.2f})")
        
        return "\n".join(context_parts)
    
    def _determine_severity(self, condition: str, score: float) -> str:
        """Determine severity level for a condition"""
        if condition not in self.condition_definitions:
            return 'unknown'
        
        thresholds = self.condition_definitions[condition]['severity_thresholds']
        
        if score >= thresholds['severe']:
            return 'severe'
        elif score >= thresholds['moderate']:
            return 'moderate'
        elif score >= thresholds['mild']:
            return 'mild'
        else:
            return 'minimal'
    
    def _determine_overall_risk(self, condition_scores: Dict) -> str:
        """FIXED: Determine overall risk level with self_harm priority"""
        if not condition_scores:
            return 'minimal'
        
        # CRITICAL: Self-harm gets immediate high risk
        self_harm_score = condition_scores.get('self_harm', 0)
        if self_harm_score > 0.3:
            return 'high'
        elif self_harm_score > 0.1:
            return 'moderate'
        
        # Regular risk assessment for other conditions
        max_score = max(condition_scores.values())
        
        if max_score >= 0.7:
            return 'high'
        elif max_score >= 0.4:
            return 'moderate'
        elif max_score >= 0.2:
            return 'low'
        else:
            return 'minimal'
    
    def _generate_recommendations(self, top_conditions: List) -> List[str]:
        """FIXED: Generate recommendations with self_harm priority"""
        recommendations = []
        
        # CRITICAL: Self-harm gets immediate crisis recommendations
        has_self_harm = any(cond['condition'] == 'self_harm' for cond in top_conditions)
        
        if has_self_harm:
            recommendations.extend([
                "🚨 IMMEDIATE: Contact crisis helpline - National Suicide Prevention Lifeline: 988",
                "🚨 URGENT: Go to nearest emergency room if in immediate danger",
                "🚨 CRITICAL: Remove any means of self-harm from your environment",
                "🚨 SUPPORT: Stay with trusted friends or family members",
                "🚨 PROFESSIONAL: Contact a mental health professional immediately"
            ])
            return recommendations[:5]  # Return only crisis recommendations
        
        # Regular recommendations for other conditions
        recommendations = [
            "Consider speaking with a mental health professional for a comprehensive evaluation",
            "Practice regular self-care activities that you enjoy",
            "Maintain a consistent sleep schedule and healthy routine"
        ]
        
        if not top_conditions:
            return recommendations
        
        condition_specific_recs = {
            'depression': [
                "Engage in regular physical exercise, even light walking can help",
                "Connect with supportive friends, family, or support groups",
                "Consider mindfulness or meditation practices"
            ],
            'anxiety': [
                "Practice deep breathing exercises and relaxation techniques",
                "Limit caffeine and alcohol consumption",
                "Try progressive muscle relaxation or yoga"
            ],
            'stress': [
                "Identify and address sources of stress where possible",
                "Practice time management and prioritization skills",
                "Take regular breaks and ensure adequate rest"
            ],
            'ptsd': [
                "Consider trauma-informed therapy approaches",
                "Develop a strong support network",
                "Practice grounding techniques during difficult moments"
            ],
            'bipolar': [
                "Maintain a mood diary to track patterns",
                "Stick to regular medication routines if prescribed",
                "Develop a crisis plan with your healthcare provider"
            ]
        }
        
        # Add specific recommendations for top conditions
        for condition_data in top_conditions:
            condition = condition_data['condition']
            if condition in condition_specific_recs:
                recommendations.extend(condition_specific_recs[condition][:2])
        
        return recommendations[:7]
    
    def determine_current_mood(self, analysis_results: Dict) -> Dict:
        """FIXED: Determine current mood with self_harm override"""
        try:
            # CRITICAL: Check for self-harm first
            text_analysis = analysis_results.get('individual_analyses', {}).get('text_analysis', {})
            if text_analysis:
                mental_health_indicators = text_analysis.get('mental_health_indicators', {})
                if 'self_harm' in mental_health_indicators:
                    return {
                        'current_mood': 'Crisis',
                        'confidence': mental_health_indicators['self_harm']['confidence'] / 100,
                        'dominant_emotion': 'sadness',
                        'mood_description': 'CRISIS STATE - Immediate help needed'
                    }
            
            # Regular mood determination
            emotion_analysis = self.combine_emotion_scores(analysis_results)
            dominant_emotion = emotion_analysis['dominant_emotion']
            confidence = emotion_analysis['confidence']
            
            # Map emotions to mood categories
            mood_mapping = {
                'joy': 'Happy', 'happiness': 'Happy', 'happy': 'Happy',
                'sadness': 'Sad', 'sad': 'Sad',
                'anger': 'Angry', 'angry': 'Angry',
                'fear': 'Anxious', 'anxiety': 'Anxious',
                'surprise': 'Surprised', 'surprised': 'Surprised',
                'disgust': 'Disgusted',
                'neutral': 'Neutral', 'calm': 'Calm',
                'excited': 'Excited', 'tired': 'Tired'
            }
            
            mood = mood_mapping.get(dominant_emotion.lower(), 'Neutral')
            
            return {
                'current_mood': mood,
                'confidence': confidence,
                'dominant_emotion': dominant_emotion,
                'mood_description': self._get_mood_description(mood, confidence)
            }
            
        except Exception as e:
            logger.error(f"Mood determination failed: {str(e)}")
            return {
                'current_mood': 'Neutral',
                'confidence': 1.0,
                'dominant_emotion': 'neutral',
                'mood_description': 'Unable to determine mood'
            }
    
    def _get_mood_description(self, mood: str, confidence: float) -> str:
        """Get description for current mood"""
        descriptions = {
            'Happy': 'Feeling positive and content',
            'Sad': 'Experiencing sadness or low mood',
            'Angry': 'Feeling frustrated or irritated',
            'Anxious': 'Experiencing worry or nervousness',
            'Surprised': 'Feeling caught off-guard or amazed',
            'Disgusted': 'Feeling repulsion or strong dislike',
            'Neutral': 'In a balanced, calm state',
            'Calm': 'Feeling peaceful and relaxed',
            'Excited': 'Feeling energetic and enthusiastic',
            'Tired': 'Feeling fatigued or weary',
            'Crisis': 'In a mental health crisis requiring immediate attention'
        }
        
        base_description = descriptions.get(mood, 'Current emotional state')
        confidence_text = "high" if confidence > 0.7 else "moderate" if confidence > 0.4 else "low"
        
        return f"{base_description} (confidence: {confidence_text})"
    
    def comprehensive_analysis(self, 
                             text: Optional[str] = None,
                             audio_file: Optional[str] = None,
                             video_frame: Optional[np.ndarray] = None) -> Dict:
        """FIXED: Perform comprehensive mental health analysis with self_harm priority"""
        try:
            # Analyze all inputs
            analysis_results = self.analyze_all_inputs(text, audio_file, video_frame)
            
            # Generate diagnosis
            diagnosis = self.generate_diagnosis(analysis_results)
            
            # Determine mood
            mood = self.determine_current_mood(analysis_results)
            
            # FIXED: Enhanced summary with self_harm priority
            primary_concern = 'None detected'
            confidence = 0
            
            if diagnosis['top_conditions']:
                # Check for self_harm first
                self_harm_condition = next((cond for cond in diagnosis['top_conditions'] if cond['condition'] == 'self_harm'), None)
                
                if self_harm_condition:
                    primary_concern = 'self_harm'
                    confidence = self_harm_condition['confidence_percentage']
                else:
                    primary_concern = diagnosis['top_conditions'][0]['condition']
                    confidence = diagnosis['top_conditions'][0]['confidence_percentage']
            
            return {
                'individual_analyses': analysis_results,
                'diagnosis': diagnosis,
                'current_mood': mood,
                'summary': {
                    'primary_concern': primary_concern,
                    'confidence': confidence,
                    'mood': mood['current_mood'],
                    'risk_level': diagnosis['overall_risk_level'],
                    'needs_attention': diagnosis['overall_risk_level'] in ['moderate', 'high'] or primary_concern == 'self_harm'
                },
                'timestamp': datetime.now().isoformat(),
                'analysis_successful': True
            }
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {str(e)}")
            return self._default_comprehensive_analysis(str(e))
    
    def _default_diagnosis(self) -> Dict:
        """Return default diagnosis structure"""
        return {
            'top_conditions': [],
            'emotion_analysis': {'combined_emotions': {'neutral': 1.0}, 'dominant_emotion': 'neutral', 'confidence': 1.0},
            'ai_analysis': 'Analysis unavailable. Please consult with a mental health professional.',
            'overall_risk_level': 'minimal',
            'recommendations': ['Consider speaking with a mental health professional'],
            'timestamp': datetime.now().isoformat()
        }
    
    def _default_comprehensive_analysis(self, error: str) -> Dict:
        """Return default comprehensive analysis"""
        return {
            'individual_analyses': {},
            'diagnosis': self._default_diagnosis(),
            'current_mood': {'current_mood': 'Neutral', 'confidence': 1.0, 'mood_description': 'Unable to determine'},
            'summary': {
                'primary_concern': 'Analysis error',
                'confidence': 0,
                'mood': 'Neutral',
                'risk_level': 'minimal',
                'needs_attention': False
            },
            'error': error,
            'timestamp': datetime.now().isoformat(),
            'analysis_successful': False
        }

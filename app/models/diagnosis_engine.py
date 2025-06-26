from .text_analyzer import TextAnalyzer
from .audio_analyzer import AudioAnalyzer
from .video_analyzer import VideoAnalyzer
import openai
from typing import Dict, List, Optional, Any
import json
import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class DiagnosisEngine:
    def __init__(self, openai_api_key: str):
        """Initialize diagnosis engine with all analyzers"""
        try:
            self.text_analyzer = TextAnalyzer()
            self.audio_analyzer = AudioAnalyzer()
            self.video_analyzer = VideoAnalyzer()
            
            # Set OpenAI API key
            openai.api_key = openai_api_key
            
            # Mental health condition definitions
            self.condition_definitions = {
                'depression': {
                    'description': 'Persistent feelings of sadness, hopelessness, and loss of interest',
                    'symptoms': ['persistent sadness', 'loss of interest', 'fatigue', 'sleep disturbances', 'appetite changes'],
                    'severity_thresholds': {'mild': 0.3, 'moderate': 0.5, 'severe': 0.7}
                },
                'anxiety': {
                    'description': 'Excessive worry, fear, and physical symptoms of stress',
                    'symptoms': ['excessive worry', 'restlessness', 'fatigue', 'difficulty concentrating', 'physical tension'],
                    'severity_thresholds': {'mild': 0.25, 'moderate': 0.45, 'severe': 0.65}
                },
                'stress': {
                    'description': 'Response to overwhelming demands or pressures',
                    'symptoms': ['feeling overwhelmed', 'irritability', 'muscle tension', 'headaches', 'sleep problems'],
                    'severity_thresholds': {'mild': 0.2, 'moderate': 0.4, 'severe': 0.6}
                },
                'ptsd': {
                    'description': 'Response to traumatic events with recurring symptoms',
                    'symptoms': ['flashbacks', 'nightmares', 'avoidance', 'hypervigilance', 'emotional numbing'],
                    'severity_thresholds': {'mild': 0.35, 'moderate': 0.55, 'severe': 0.75}
                },
                'bipolar': {
                    'description': 'Alternating periods of depression and mania',
                    'symptoms': ['mood swings', 'energy fluctuations', 'sleep changes', 'impulsivity', 'racing thoughts'],
                    'severity_thresholds': {'mild': 0.4, 'moderate': 0.6, 'severe': 0.8}
                }
            }
            
            # Weight assignments for different input types
            self.modality_weights = {
                'text': 0.4,
                'audio': 0.35,
                'video': 0.25
            }
            
            logger.info("Diagnosis engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing diagnosis engine: {str(e)}")
            raise
    
    def analyze_all_inputs(self, 
                          text: Optional[str] = None,
                          audio_file: Optional[str] = None,
                          video_frame: Optional[np.ndarray] = None) -> Dict:
        """Analyze all provided inputs"""
        
        analysis_results = {
            'text_analysis': None,
            'audio_analysis': None,
            'video_analysis': None,
            'timestamp': datetime.now().isoformat()
        }
        
        # Text analysis
        if text and len(text.strip()) > 0:
            try:
                analysis_results['text_analysis'] = self.text_analyzer.analyze_text(text)
                logger.info("Text analysis completed")
            except Exception as e:
                logger.error(f"Text analysis failed: {str(e)}")
                analysis_results['text_analysis'] = {'error': str(e)}
        
        # Audio analysis
        if audio_file:
            try:
                analysis_results['audio_analysis'] = self.audio_analyzer.analyze_audio(audio_file)
                logger.info("Audio analysis completed")
            except Exception as e:
                logger.error(f"Audio analysis failed: {str(e)}")
                analysis_results['audio_analysis'] = {'error': str(e)}
        
        # Video analysis
        if video_frame is not None:
            try:
                analysis_results['video_analysis'] = self.video_analyzer.analyze_frame(video_frame)
                logger.info("Video frame analysis completed")
            except Exception as e:
                logger.error(f"Video analysis failed: {str(e)}")
                analysis_results['video_analysis'] = {'error': str(e)}
        
        return analysis_results
    
    def combine_emotion_scores(self, analysis_results: Dict) -> Dict:
        """Combine emotion scores from all modalities"""
        combined_emotions = {}
        total_weight = 0
        
        # Text emotions
        if analysis_results.get('text_analysis') and 'emotions' in analysis_results['text_analysis']:
            text_emotions = analysis_results['text_analysis']['emotions']['all_scores']
            weight = self.modality_weights['text']
            
            for emotion, score in text_emotions.items():
                if emotion not in combined_emotions:
                    combined_emotions[emotion] = 0
                combined_emotions[emotion] += score * weight
            total_weight += weight
        
        # Audio emotions
        if analysis_results.get('audio_analysis') and 'emotions' in analysis_results['audio_analysis']:
            audio_emotions = analysis_results['audio_analysis']['emotions']['emotions']
            weight = self.modality_weights['audio']
            
            for emotion, score in audio_emotions.items():
                if emotion not in combined_emotions:
                    combined_emotions[emotion] = 0
                combined_emotions[emotion] += score * weight
            total_weight += weight
        
        # Video emotions
        if analysis_results.get('video_analysis') and 'emotions' in analysis_results['video_analysis']:
            video_emotions = analysis_results['video_analysis']['emotions']
            weight = self.modality_weights['video']
            
            for emotion, score in video_emotions.items():
                if emotion not in combined_emotions:
                    combined_emotions[emotion] = 0
                combined_emotions[emotion] += score * weight
            total_weight += weight
        
        # Normalize scores
        if total_weight > 0:
            combined_emotions = {
                emotion: score / total_weight 
                for emotion, score in combined_emotions.items()
            }
        else:
            combined_emotions = {'neutral': 1.0}
        
        # Get dominant emotion
        dominant_emotion = max(combined_emotions, key=combined_emotions.get)
        
        return {
            'combined_emotions': combined_emotions,
            'dominant_emotion': dominant_emotion,
            'confidence': combined_emotions[dominant_emotion],
            'total_weight': total_weight
        }
    
    def calculate_condition_scores(self, analysis_results: Dict) -> Dict:
        """Calculate scores for different mental health conditions"""
        condition_scores = {}
        
        # Get combined risk scores
        text_risk = 0
        audio_risk = 0
        video_risk = 0
        
        if analysis_results.get('text_analysis'):
            text_risk = analysis_results['text_analysis'].get('risk_score', 0)
        
        if analysis_results.get('audio_analysis'):
            audio_risk = analysis_results['audio_analysis'].get('risk_score', 0)
        
        if analysis_results.get('video_analysis'):
            video_risk = analysis_results['video_analysis'].get('risk_score', 0)
        
        # Combined risk score
        total_weight = 0
        combined_risk = 0
        
        if text_risk > 0:
            combined_risk += text_risk * self.modality_weights['text']
            total_weight += self.modality_weights['text']
        
        if audio_risk > 0:
            combined_risk += audio_risk * self.modality_weights['audio']
            total_weight += self.modality_weights['audio']
        
        if video_risk > 0:
            combined_risk += video_risk * self.modality_weights['video']
            total_weight += self.modality_weights['video']
        
        if total_weight > 0:
            combined_risk = combined_risk / total_weight
        
        # Get emotion analysis
        emotion_analysis = self.combine_emotion_scores(analysis_results)
        dominant_emotion = emotion_analysis['dominant_emotion']
        
        # Calculate condition-specific scores
        for condition in self.condition_definitions:
            score = self._calculate_condition_specific_score(
                condition, analysis_results, combined_risk, dominant_emotion
            )
            condition_scores[condition] = score
        
        return condition_scores
    
    def _calculate_condition_specific_score(self, condition: str, analysis_results: Dict, 
                                          base_risk: float, dominant_emotion: str) -> float:
        """Calculate score for a specific condition"""
        score = base_risk * 0.5  # Base score from general risk
        
        # Emotion-based scoring
        emotion_weights = {
            'depression': {
                'sadness': 0.4, 'anger': 0.2, 'fear': 0.1, 'disgust': 0.1,
                'joy': -0.2, 'surprise': 0.0, 'neutral': 0.1, 'sad': 0.4, 'tired': 0.3
            },
            'anxiety': {
                'fear': 0.4, 'surprise': 0.2, 'anger': 0.2, 'sadness': 0.1,
                'joy': -0.1, 'disgust': 0.1, 'neutral': 0.0, 'worried': 0.4
            },
            'stress': {
                'anger': 0.3, 'fear': 0.2, 'sadness': 0.2, 'disgust': 0.1,
                'joy': -0.1, 'surprise': 0.1, 'neutral': 0.1, 'tired': 0.2
            },
            'ptsd': {
                'fear': 0.4, 'anger': 0.3, 'sadness': 0.2, 'surprise': 0.1,
                'joy': -0.1, 'disgust': 0.0, 'neutral': 0.0
            },
            'bipolar': {
                'joy': 0.2, 'anger': 0.2, 'sadness': 0.2, 'surprise': 0.2,
                'fear': 0.1, 'disgust': 0.1, 'neutral': 0.0, 'happy': 0.2
            }
        }
        
        if condition in emotion_weights:
            emotion_score = emotion_weights[condition].get(dominant_emotion, 0)
            score += emotion_score * 0.3
        
        # Text-specific indicators
        if analysis_results.get('text_analysis') and 'mental_health_indicators' in analysis_results['text_analysis']:
            indicators = analysis_results['text_analysis']['mental_health_indicators']
            if condition in indicators:
                score += indicators[condition]['severity_score'] * 0.2
        
        return min(max(score, 0.0), 1.0)
    
    def generate_diagnosis(self, analysis_results: Dict) -> Dict:
        """Generate comprehensive diagnosis"""
        try:
            # Calculate condition scores
            condition_scores = self.calculate_condition_scores(analysis_results)
            
            # Get top 2 conditions
            sorted_conditions = sorted(condition_scores.items(), key=lambda x: x[1], reverse=True)
            top_conditions = sorted_conditions[:2]
            
            # Get emotion analysis
            emotion_analysis = self.combine_emotion_scores(analysis_results)
            
            # Generate AI-powered analysis
            ai_analysis = self._generate_ai_diagnosis(analysis_results, top_conditions, emotion_analysis)
            
            return {
                'top_conditions': [
                    {
                        'condition': condition,
                        'score': score,
                        'confidence_percentage': int(score * 100),
                        'severity': self._determine_severity(condition, score),
                        'description': self.condition_definitions[condition]['description']
                    }
                    for condition, score in top_conditions
                ],
                'emotion_analysis': emotion_analysis,
                'ai_analysis': ai_analysis,
                'overall_risk_level': self._determine_overall_risk(condition_scores),
                'recommendations': self._generate_recommendations(top_conditions),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Diagnosis generation failed: {str(e)}")
            return self._default_diagnosis()
    
    def _generate_ai_diagnosis(self, analysis_results: Dict, top_conditions: List, emotion_analysis: Dict) -> str:
        """Generate AI-powered diagnosis using OpenAI"""
        try:
            # Prepare context for AI
            context = self._prepare_ai_context(analysis_results, top_conditions, emotion_analysis)
            
            prompt = f"""
            As a mental health assessment AI, analyze the following multi-modal data and provide insights:

            {context}

            Please provide:
            1. A brief assessment of the person's current mental state
            2. Explanation of the top concerns identified
            3. Confidence level in the assessment
            4. Important considerations or limitations

            Be empathetic, professional, and always recommend professional consultation for serious concerns.
            Respond in a structured, easy-to-understand format.
                        """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a compassionate mental health assessment AI that provides supportive, professional insights while always encouraging professional consultation."
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
                context_parts.append(f"Mental health keywords found: {', '.join(indicators)}")
        
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
            conditions_text = ", ".join([f"{cond} ({score:.2f})" for cond, score in top_conditions])
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
        """Determine overall risk level"""
        max_score = max(condition_scores.values()) if condition_scores else 0
        
        if max_score >= 0.7:
            return 'high'
        elif max_score >= 0.4:
            return 'moderate'
        elif max_score >= 0.2:
            return 'low'
        else:
            return 'minimal'
    
    def _generate_recommendations(self, top_conditions: List) -> List[str]:
        """Generate recommendations based on top conditions"""
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
        for condition, _ in top_conditions:
            if condition in condition_specific_recs:
                recommendations.extend(condition_specific_recs[condition][:2])
        
        return recommendations[:7]  # Limit to 7 recommendations
    
    def determine_current_mood(self, analysis_results: Dict) -> Dict:
        """Determine current mood from analysis"""
        emotion_analysis = self.combine_emotion_scores(analysis_results)
        dominant_emotion = emotion_analysis['dominant_emotion']
        confidence = emotion_analysis['confidence']
        
        # Map emotions to mood categories
        mood_mapping = {
            'joy': 'Happy',
            'happiness': 'Happy',
            'happy': 'Happy',
            'sadness': 'Sad',
            'sad': 'Sad',
            'anger': 'Angry',
            'angry': 'Angry',
            'fear': 'Anxious',
            'anxiety': 'Anxious',
            'surprise': 'Surprised',
            'surprised': 'Surprised',
            'disgust': 'Disgusted',
            'neutral': 'Neutral',
            'calm': 'Calm',
            'excited': 'Excited',
            'tired': 'Tired'
        }
        
        mood = mood_mapping.get(dominant_emotion.lower(), 'Neutral')
        
        return {
            'current_mood': mood,
            'confidence': confidence,
            'dominant_emotion': dominant_emotion,
            'mood_description': self._get_mood_description(mood, confidence)
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
            'Tired': 'Feeling fatigued or weary'
        }
        
        base_description = descriptions.get(mood, 'Current emotional state')
        confidence_text = "high" if confidence > 0.7 else "moderate" if confidence > 0.4 else "low"
        
        return f"{base_description} (confidence: {confidence_text})"
    
    def comprehensive_analysis(self, 
                             text: Optional[str] = None,
                             audio_file: Optional[str] = None,
                             video_frame: Optional[np.ndarray] = None) -> Dict:
        """Perform comprehensive mental health analysis"""
        try:
            # Analyze all inputs
            analysis_results = self.analyze_all_inputs(text, audio_file, video_frame)
            
            # Generate diagnosis
            diagnosis = self.generate_diagnosis(analysis_results)
            
            # Determine mood
            mood = self.determine_current_mood(analysis_results)
            
            return {
                'individual_analyses': analysis_results,
                'diagnosis': diagnosis,
                'current_mood': mood,
                'summary': {
                    'primary_concern': diagnosis['top_conditions'][0]['condition'] if diagnosis['top_conditions'] else 'None detected',
                    'confidence': diagnosis['top_conditions'][0]['confidence_percentage'] if diagnosis['top_conditions'] else 0,
                    'mood': mood['current_mood'],
                    'risk_level': diagnosis['overall_risk_level'],
                    'needs_attention': diagnosis['overall_risk_level'] in ['moderate', 'high']
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


import librosa
import numpy as np
import soundfile as sf
from transformers import pipeline
import torch
import tempfile
import os
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class AudioAnalyzer:
    def __init__(self):
        """Initialize audio analysis models"""
        try:
            # Speech-to-text model
            self.speech_to_text = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-base",
                return_timestamps=True
            )
            
            # Emotion recognition model
            try:
                self.emotion_classifier = pipeline(
                    "audio-classification",
                    model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
                )
            except:
                logger.warning("Primary emotion model failed, using fallback")
                self.emotion_classifier = None
            
            logger.info("Audio analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing audio analyzer: {str(e)}")
            raise
    
    def load_audio(self, audio_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
        """Load audio file and convert to specified sample rate"""
        try:
            audio, original_sr = librosa.load(audio_path, sr=sr)
            return audio, sr
        except Exception as e:
            logger.error(f"Failed to load audio: {str(e)}")
            raise
    
    def extract_audio_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract comprehensive audio features"""
        try:
            features = {}
            
            # Basic statistics
            features['duration'] = len(audio) / sr
            features['sample_rate'] = sr
            features['rms_energy'] = float(np.sqrt(np.mean(audio**2)))
            features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            
            # MFCC features (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i]))
            
            # Pitch analysis
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitch_values = pitches[pitches > 0]
            if len(pitch_values) > 0:
                features['pitch_mean'] = float(np.mean(pitch_values))
                features['pitch_std'] = float(np.std(pitch_values))
                features['pitch_min'] = float(np.min(pitch_values))
                features['pitch_max'] = float(np.max(pitch_values))
            else:
                features.update({
                    'pitch_mean': 0.0, 'pitch_std': 0.0,
                    'pitch_min': 0.0, 'pitch_max': 0.0
                })
            
            # Tempo and rhythm
            try:
                tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
                features['tempo'] = float(tempo)
                features['beat_count'] = len(beats)
            except:
                features['tempo'] = 0.0
                features['beat_count'] = 0
            
            # Chroma features (pitch class profile)
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features['chroma_mean'] = float(np.mean(chroma))
            features['chroma_std'] = float(np.std(chroma))
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return {'error': str(e)}
    
    def detect_voice_activity(self, audio: np.ndarray, sr: int) -> Dict:
        """Simple voice activity detection"""
        try:
            # Frame-based energy analysis
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.01 * sr)     # 10ms hop
            
            # Calculate frame energy
            frames = librosa.util.frame(audio, frame_length=frame_length, 
                                      hop_length=hop_length, axis=0)
            energy = np.sum(frames**2, axis=0)
            
            # Threshold-based VAD
            energy_threshold = np.mean(energy) * 0.1
            voice_frames = energy > energy_threshold
            
            # Calculate speech statistics
            total_frames = len(voice_frames)
            speech_frames = np.sum(voice_frames)
            speech_ratio = speech_frames / total_frames if total_frames > 0 else 0
            
            # Find speech segments
            segments = []
            in_speech = False
            start_frame = 0
            
            for i, is_speech in enumerate(voice_frames):
                if is_speech and not in_speech:
                    start_frame = i
                    in_speech = True
                elif not is_speech and in_speech:
                    segments.append((start_frame * hop_length / sr, i * hop_length / sr))
                    in_speech = False
            
            if in_speech:  # Close last segment
                segments.append((start_frame * hop_length / sr, len(voice_frames) * hop_length / sr))
            
            return {
                'speech_ratio': float(speech_ratio),
                'total_duration': len(audio) / sr,
                'speech_duration': float(speech_ratio * len(audio) / sr),
                'speech_segments': segments,
                'num_segments': len(segments)
            }
            
        except Exception as e:
            logger.error(f"VAD failed: {str(e)}")
            return {'speech_ratio': 0.5, 'total_duration': len(audio) / sr, 
                   'speech_duration': len(audio) / sr * 0.5, 'speech_segments': [], 'num_segments': 0}
    
    def transcribe_audio(self, audio_path: str) -> Dict:
        """Transcribe audio to text"""
        try:
            result = self.speech_to_text(audio_path)
            
            if isinstance(result, dict) and 'text' in result:
                transcription = result['text']
                timestamps = result.get('chunks', [])
            elif isinstance(result, str):
                transcription = result
                timestamps = []
            else:
                transcription = str(result)
                timestamps = []
            
            return {
                'transcription': transcription,
                'timestamps': timestamps,
                'confidence': 1.0,  # Whisper doesn't provide confidence scores
                'language': 'auto-detected'
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            return {
                'transcription': '',
                'timestamps': [],
                'confidence': 0.0,
                'language': 'unknown',
                'error': str(e)
            }
    
    def analyze_emotion_from_audio(self, audio_path: str) -> Dict:
        """Analyze emotions from audio"""
        try:
            if self.emotion_classifier is None:
                return self._fallback_emotion_analysis(audio_path)
            
            # Use the emotion classifier
            result = self.emotion_classifier(audio_path)
            
            # Normalize results
            emotions = {}
            if isinstance(result, list):
                for item in result:
                    emotions[item['label']] = item['score']
            
            # Get top emotion
            if emotions:
                top_emotion = max(emotions, key=emotions.get)
                confidence = emotions[top_emotion]
            else:
                top_emotion = 'neutral'
                confidence = 1.0
                emotions = {'neutral': 1.0}
            
            return {
                'emotions': emotions,
                'dominant_emotion': top_emotion,
                'confidence': confidence,
                'model_used': 'wav2vec2-emotion'
            }
            
        except Exception as e:
            logger.error(f"Audio emotion analysis failed: {str(e)}")
            return self._fallback_emotion_analysis(audio_path)
    
    def _fallback_emotion_analysis(self, audio_path: str) -> Dict:
        """Fallback emotion analysis based on audio features"""
        try:
            audio, sr = self.load_audio(audio_path)
            features = self.extract_audio_features(audio, sr)
            
            # Simple rule-based emotion detection
            emotions = {'neutral': 0.5, 'calm': 0.3, 'unknown': 0.2}
            
            # Basic heuristics based on audio features
            if features.get('rms_energy', 0) > 0.1:
                emotions['excited'] = 0.4
                emotions['angry'] = 0.3
                emotions['neutral'] = 0.3
            elif features.get('rms_energy', 0) < 0.05:
                emotions['sad'] = 0.4
                emotions['calm'] = 0.4
                emotions['neutral'] = 0.2
            
            top_emotion = max(emotions, key=emotions.get)
            
            return {
                'emotions': emotions,
                'dominant_emotion': top_emotion,
                'confidence': emotions[top_emotion],
                'model_used': 'rule-based-fallback'
            }
            
        except Exception as e:
            logger.error(f"Fallback emotion analysis failed: {str(e)}")
            return {
                'emotions': {'neutral': 1.0},
                'dominant_emotion': 'neutral',
                'confidence': 1.0,
                'model_used': 'default'
            }
    
    def calculate_audio_risk_score(self, features: Dict, emotions: Dict) -> float:
        """Calculate risk score based on audio analysis"""
        try:
            risk_score = 0.0
            
            # Energy-based indicators
            energy = features.get('rms_energy', 0)
            if energy < 0.02:  # Very low energy might indicate depression
                risk_score += 0.2
            elif energy > 0.15:  # Very high energy might indicate mania/anxiety
                risk_score += 0.15
            
            # Pitch-based indicators
            pitch_std = features.get('pitch_std', 0)
            if pitch_std > 50:  # High pitch variation might indicate emotional distress
                risk_score += 0.15
            
            # Speaking rate indicators
            speech_ratio = features.get('speech_ratio', 0.5)
            if speech_ratio < 0.3:  # Very little speech might indicate withdrawal
                risk_score += 0.2
            
            # Emotion-based risk
            emotion_risks = {
                'sad': 0.3, 'angry': 0.25, 'fear': 0.3, 'disgust': 0.2,
                'surprised': 0.1, 'happy': -0.1, 'calm': -0.1, 'neutral': 0.0
            }
            
            for emotion, score in emotions.get('emotions', {}).items():
                emotion_risk = emotion_risks.get(emotion, 0.0)
                risk_score += emotion_risk * score
            
            return min(max(risk_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Risk calculation failed: {str(e)}")
            return 0.0
    
    def analyze_audio(self, audio_file_path: str) -> Dict:
        """Comprehensive audio analysis"""
        if not os.path.exists(audio_file_path):
            return self._empty_audio_analysis()
        
        try:
            # Load audio
            audio, sr = self.load_audio(audio_file_path)
            
            # Extract features
            features = self.extract_audio_features(audio, sr)
            
            # Voice activity detection
            vad_results = self.detect_voice_activity(audio, sr)
            
            # Speech-to-text
            transcription = self.transcribe_audio(audio_file_path)
            
            # Emotion analysis
            emotions = self.analyze_emotion_from_audio(audio_file_path)
            
            # Risk assessment
            risk_score = self.calculate_audio_risk_score({**features, **vad_results}, emotions)
            
            return {
                'file_path': audio_file_path,
                'audio_features': features,
                'voice_activity': vad_results,
                'transcription': transcription,
                'emotions': emotions,
                'risk_score': risk_score,
                'analysis_success': True
            }
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {str(e)}")
            return {
                **self._empty_audio_analysis(),
                'error': str(e),
                'analysis_success': False
            }
    
    def _empty_audio_analysis(self) -> Dict:
        """Return empty audio analysis structure"""
        return {
            'file_path': '',
            'audio_features': {},
            'voice_activity': {'speech_ratio': 0.0, 'total_duration': 0.0},
            'transcription': {'transcription': '', 'confidence': 0.0},
            'emotions': {'emotions': {'neutral': 1.0}, 'dominant_emotion': 'neutral', 'confidence': 1.0},
            'risk_score': 0.0,
            'analysis_success': False
        }

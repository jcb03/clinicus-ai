import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from transformers import pipeline
import logging
from typing import Dict, List, Optional, Tuple
import time

logger = logging.getLogger(__name__)

class VideoAnalyzer:
    def __init__(self):
        """Initialize video analysis models using MediaPipe"""
        try:
            # Initialize MediaPipe Face Detection
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
            
            # Initialize MediaPipe Face Mesh for landmarks
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            
            # Face emotion classification (fallback to rule-based if model fails)
            try:
                self.face_emotion_classifier = pipeline(
                    "image-classification",
                    model="trpakov/vit-face-expression"
                )
            except:
                logger.warning("Face emotion model not available, using rule-based analysis")
                self.face_emotion_classifier = None
            
            # Initialize emotion tracking variables
            self.emotion_history = []
            
            logger.info("Video analyzer with MediaPipe initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing video analyzer: {str(e)}")
            raise
    
    def detect_faces_mediapipe(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces using MediaPipe"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            
            faces = []
            if results.detections:
                h, w, _ = frame.shape
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    faces.append({
                        'bbox': (x, y, width, height),
                        'confidence': detection.score[0] if detection.score else 0.0
                    })
            
            return faces
            
        except Exception as e:
            logger.error(f"MediaPipe face detection failed: {str(e)}")
            return []
    
    def extract_face_landmarks(self, frame: np.ndarray) -> Dict:
        """Extract facial landmarks using MediaPipe"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            landmarks_data = {
                'landmarks_detected': False,
                'landmark_count': 0,
                'face_landmarks': []
            }
            
            if results.multi_face_landmarks:
                landmarks_data['landmarks_detected'] = True
                for face_landmarks in results.multi_face_landmarks:
                    landmarks_points = []
                    for landmark in face_landmarks.landmark:
                        landmarks_points.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        })
                    
                    landmarks_data['face_landmarks'].append(landmarks_points)
                    landmarks_data['landmark_count'] = len(landmarks_points)
            
            return landmarks_data
            
        except Exception as e:
            logger.error(f"Landmark extraction failed: {str(e)}")
            return {
                'landmarks_detected': False,
                'landmark_count': 0,
                'face_landmarks': []
            }
    
    def analyze_facial_geometry(self, landmarks: List[Dict]) -> Dict:
        """Analyze facial geometry from landmarks"""
        try:
            if not landmarks:
                return {'symmetry_score': 0.5, 'face_ratio': 1.0, 'mouth_openness': 0.0}
            
            # Convert landmarks to numpy array
            points = np.array([(lm['x'], lm['y']) for lm in landmarks])
            
            # Calculate face width and height
            face_width = np.max(points[:, 0]) - np.min(points[:, 0])
            face_height = np.max(points[:, 1]) - np.min(points[:, 1])
            face_ratio = face_width / face_height if face_height > 0 else 1.0
            
            # Simple symmetry analysis
            center_x = np.mean(points[:, 0])
            left_points = points[points[:, 0] < center_x]
            right_points = points[points[:, 0] > center_x]
            
            if len(left_points) > 0 and len(right_points) > 0:
                left_var = np.var(left_points[:, 1])
                right_var = np.var(right_points[:, 1])
                symmetry_score = 1.0 - abs(left_var - right_var) / max(left_var + right_var, 0.001)
            else:
                symmetry_score = 0.5
            
            # Mouth analysis (simplified)
            mouth_landmarks = landmarks[61:81] if len(landmarks) > 81 else []
            if mouth_landmarks:
                mouth_points = np.array([(lm['x'], lm['y']) for lm in mouth_landmarks])
                mouth_height = np.max(mouth_points[:, 1]) - np.min(mouth_points[:, 1])
                mouth_openness = mouth_height / face_height if face_height > 0 else 0.0
            else:
                mouth_openness = 0.0
            
            return {
                'symmetry_score': float(symmetry_score),
                'face_ratio': float(face_ratio),
                'mouth_openness': float(mouth_openness)
            }
            
        except Exception as e:
            logger.error(f"Facial geometry analysis failed: {str(e)}")
            return {'symmetry_score': 0.5, 'face_ratio': 1.0, 'mouth_openness': 0.0}
    
    def classify_facial_emotion(self, face_roi: np.ndarray) -> Dict:
        """Classify facial emotions"""
        try:
            if self.face_emotion_classifier is not None:
                # Convert BGR to RGB for PIL
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)
                
                # Classify emotions
                results = self.face_emotion_classifier(face_pil)
                
                # Process results
                emotions = {}
                for result in results:
                    emotions[result['label']] = result['score']
                
                # Get dominant emotion
                if emotions:
                    dominant_emotion = max(emotions, key=emotions.get)
                    confidence = emotions[dominant_emotion]
                else:
                    dominant_emotion = 'neutral'
                    confidence = 1.0
                    emotions = {'neutral': 1.0}
                
                return {
                    'emotions': emotions,
                    'dominant_emotion': dominant_emotion,
                    'confidence': confidence,
                    'method': 'transformer_model'
                }
            
            else:
                # Fallback to rule-based emotion detection
                return self._rule_based_emotion_analysis(face_roi)
            
        except Exception as e:
            logger.error(f"Facial emotion classification failed: {str(e)}")
            return self._rule_based_emotion_analysis(face_roi)
    
    def _rule_based_emotion_analysis(self, face_roi: np.ndarray) -> Dict:
        """Rule-based emotion analysis using basic image properties"""
        try:
            # Convert to grayscale
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Basic image statistics
            brightness = np.mean(gray_face)
            contrast = np.std(gray_face)
            
            # Simple rule-based classification
            emotions = {'neutral': 0.4}
            
            if brightness < 100:  # Dark image might indicate sadness
                emotions['sad'] = 0.3
                emotions['tired'] = 0.2
                emotions['neutral'] = 0.1
            elif brightness > 150:  # Bright image might indicate happiness
                emotions['happy'] = 0.4
                emotions['surprised'] = 0.2
                emotions['neutral'] = 0.4
            
            if contrast < 30:  # Low contrast might indicate calmness or tiredness
                emotions['calm'] = 0.3
                emotions['tired'] = 0.2
            elif contrast > 70:  # High contrast might indicate strong emotions
                emotions['angry'] = 0.2
                emotions['surprised'] = 0.2
            
            # Normalize emotions
            total_score = sum(emotions.values())
            if total_score > 0:
                emotions = {k: v/total_score for k, v in emotions.items()}
            
            dominant_emotion = max(emotions, key=emotions.get)
            
            return {
                'emotions': emotions,
                'dominant_emotion': dominant_emotion,
                'confidence': emotions[dominant_emotion],
                'method': 'rule_based'
            }
            
        except Exception as e:
            logger.error(f"Rule-based emotion analysis failed: {str(e)}")
            return {
                'emotions': {'neutral': 1.0},
                'dominant_emotion': 'neutral',
                'confidence': 1.0,
                'method': 'default'
            }
    
    def track_emotion_changes(self, current_emotion: str, timestamp: float = None) -> Dict:
        """Track emotion changes over time"""
        if timestamp is None:
            timestamp = time.time()
        
        self.emotion_history.append({
            'emotion': current_emotion,
            'timestamp': timestamp
        })
        
        # Keep only last 30 seconds of history
        current_time = time.time()
        self.emotion_history = [
            entry for entry in self.emotion_history 
            if current_time - entry['timestamp'] <= 30
        ]
        
        # Analyze patterns
        if len(self.emotion_history) < 2:
            return {'stability': 1.0, 'changes': 0, 'dominant_recent': current_emotion}
        
        # Count emotion changes
        changes = 0
        for i in range(1, len(self.emotion_history)):
            if self.emotion_history[i]['emotion'] != self.emotion_history[i-1]['emotion']:
                changes += 1
        
        # Calculate stability (fewer changes = more stable)
        stability = max(0.0, 1.0 - (changes / len(self.emotion_history)))
        
        # Find most common recent emotion
        recent_emotions = [entry['emotion'] for entry in self.emotion_history[-10:]]
        dominant_recent = max(set(recent_emotions), key=recent_emotions.count) if recent_emotions else current_emotion
        
        return {
            'stability': stability,
            'changes': changes,
            'dominant_recent': dominant_recent,
            'history_length': len(self.emotion_history)
        }
    
    def calculate_video_risk_score(self, emotions: Dict, geometry: Dict, tracking: Dict) -> float:
        """Calculate risk score based on video analysis"""
        try:
            risk_score = 0.0
            
            # Emotion-based risk
            emotion_risks = {
                'sad': 0.4, 'angry': 0.3, 'fear': 0.35, 'disgust': 0.25,
                'surprised': 0.1, 'happy': -0.1, 'neutral': 0.0, 'tired': 0.3
            }
            
            dominant_emotion = emotions.get('dominant_emotion', 'neutral')
            emotion_confidence = emotions.get('confidence', 0.0)
            emotion_risk = emotion_risks.get(dominant_emotion, 0.0)
            risk_score += emotion_risk * emotion_confidence
            
            # Facial geometry-based risk
            symmetry = geometry.get('symmetry_score', 0.5)
            if symmetry < 0.3:  # Low symmetry might indicate stress
                risk_score += 0.1
            
            mouth_openness = geometry.get('mouth_openness', 0.0)
            if mouth_openness > 0.1:  # Wide open mouth might indicate distress
                risk_score += 0.1
            
            # Emotional stability
            stability = tracking.get('stability', 1.0)
            if stability < 0.5:  # High emotional volatility
                risk_score += 0.2
            
            return min(max(risk_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Video risk calculation failed: {str(e)}")
            return 0.0
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """Analyze single video frame"""
        try:
            # Detect faces
            faces = self.detect_faces_mediapipe(frame)
            
            if not faces:
                return {
                    'faces_detected': 0,
                    'face_locations': [],
                    'emotions': {'neutral': 1.0},
                    'dominant_emotion': 'neutral',
                    'confidence': 1.0,
                    'facial_geometry': {},
                    'landmarks': {},
                    'risk_score': 0.0,
                    'tracking': {'stability': 1.0, 'changes': 0}
                }
            
            # Analyze the most confident face
            best_face = max(faces, key=lambda x: x['confidence'])
            x, y, w, h = best_face['bbox']
            
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            # Extract landmarks
            landmarks_data = self.extract_face_landmarks(frame)
            
            # Analyze facial geometry
            geometry_results = {}
            if landmarks_data['landmarks_detected'] and landmarks_data['face_landmarks']:
                geometry_results = self.analyze_facial_geometry(landmarks_data['face_landmarks'][0])
            
            # Emotion classification
            emotion_results = self.classify_facial_emotion(face_roi)
            
            # Track emotion changes
            tracking_results = self.track_emotion_changes(emotion_results['dominant_emotion'])
            
            # Calculate risk score
            risk_score = self.calculate_video_risk_score(
                emotion_results, geometry_results, tracking_results
            )
            
            return {
                'faces_detected': len(faces),
                'face_locations': [face['bbox'] for face in faces],
                'best_face_bbox': best_face['bbox'],
                'best_face_confidence': best_face['confidence'],
                'emotions': emotion_results['emotions'],
                'dominant_emotion': emotion_results['dominant_emotion'],
                'confidence': emotion_results['confidence'],
                'emotion_method': emotion_results.get('method', 'unknown'),
                'facial_geometry': geometry_results,
                'landmarks': landmarks_data,
                'tracking': tracking_results,
                'risk_score': risk_score,
                'analysis_timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Frame analysis failed: {str(e)}")
            return self._empty_frame_analysis()
    
    def analyze_image(self, image_path: str) -> Dict:
        """Analyze static image"""
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                raise ValueError(f"Cannot load image: {image_path}")
            
            return self.analyze_frame(frame)
            
        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}")
            return self._empty_frame_analysis()
    
    def _empty_frame_analysis(self) -> Dict:
        """Return empty frame analysis"""
        return {
            'faces_detected': 0,
            'face_locations': [],
            'emotions': {'neutral': 1.0},
            'dominant_emotion': 'neutral',
            'confidence': 1.0,
            'facial_geometry': {},
            'landmarks': {},
            'risk_score': 0.0,
            'tracking': {'stability': 1.0, 'changes': 0}
        }

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
import plotly.express as px
from datetime import datetime
import pandas as pd
import logging
import io
import base64

# Try to import live recording components
try:
    import pyaudio
    import wave
    import threading
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    st.warning("⚠️ Live audio recording not available. Install pyaudio: pip install pyaudio")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from models.diagnosis_engine import DiagnosisEngine
from utils.conversation_manager import ConversationManager
from utils.openai_client import OpenAIClient
from config.settings import settings

# Page configuration
st.set_page_config(
    page_title="🧠 Mental Health Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .live-recording {
        background: #fff5f5;
        border: 2px solid #dc3545;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(220, 53, 69, 0); }
        100% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0); }
    }
    
    .primary-concern-highlight {
        background: linear-gradient(90deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff6b6b;
        margin: 1rem 0;
        font-weight: bold;
    }
    
    .crisis-alert {
        background: #ffebee;
        border: 2px solid #f44336;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #d32f2f;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'recording_audio' not in st.session_state:
    st.session_state.recording_audio = False
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None

def initialize_app():
    """Initialize application components"""
    try:
        if 'diagnosis_engine' not in st.session_state:
            with st.spinner("🤖 Loading AI models..."):
                st.session_state.diagnosis_engine = DiagnosisEngine(settings.openai_api_key)
        
        if 'conversation_manager' not in st.session_state:
            st.session_state.conversation_manager = ConversationManager()
        
        if 'openai_client' not in st.session_state:
            st.session_state.openai_client = OpenAIClient(settings.openai_api_key)
        
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        
        if 'current_mood' not in st.session_state:
            st.session_state.current_mood = "Unknown"
        
        return True
        
    except Exception as e:
        st.error(f"❌ Failed to initialize: {str(e)}")
        return False

def record_audio(duration=5):
    """Record audio using pyaudio"""
    if not AUDIO_AVAILABLE:
        st.error("Audio recording not available")
        return None
    
    try:
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        
        p = pyaudio.PyAudio()
        
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)
        
        frames = []
        
        for i in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            wf = wave.open(tmp_file.name, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            return tmp_file.name
    
    except Exception as e:
        st.error(f"Recording failed: {e}")
        return None

def display_analysis_section():
    """Analysis section with live features"""
    st.header("🎯 Mental Health Analysis")
    st.write("Analyze your mental state through text, voice, and photo inputs")
    
    tab1, tab2, tab3 = st.tabs(["💬 Text Input", "🎤 Live Audio", "📹 Live Video"])
    
    text_input = None
    audio_file_path = None
    video_frame = None
    
    # Tab 1: Text Input
    with tab1:
        st.subheader("Share Your Thoughts")
        
        text_input = st.text_area(
            "How are you feeling?",
            height=150,
            placeholder="Example: 'I've been feeling really sad lately and can't sleep. I feel hopeless and don't know what to do...'",
            help="Express your thoughts and feelings openly for better analysis"
        )
        
        if text_input:
            word_count = len(text_input.split())
            st.caption(f"Word count: {word_count}")
    
    # Tab 2: Live Audio Recording
    with tab2:
        st.subheader("🎤 Live Audio Recording")
        
        if AUDIO_AVAILABLE:
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if st.button("🔴 Start Recording", disabled=st.session_state.recording_audio):
                    st.session_state.recording_audio = True
                    with st.spinner("Recording for 10 seconds..."):
                        audio_file_path = record_audio(10)
                        if audio_file_path:
                            st.session_state.audio_data = audio_file_path
                            st.success("✅ Recording completed!")
                            # Play back the recorded audio
                            with open(audio_file_path, 'rb') as f:
                                st.audio(f.read(), format='audio/wav')
                    st.session_state.recording_audio = False
            
            with col2:
                if st.session_state.get('audio_data'):
                    if st.button("🗑️ Clear Recording"):
                        if os.path.exists(st.session_state.audio_data):
                            os.unlink(st.session_state.audio_data)
                        st.session_state.audio_data = None
                        st.success("Recording cleared!")
            
            if st.session_state.recording_audio:
                st.markdown('<div class="live-recording">🎤 Recording in progress...</div>', 
                           unsafe_allow_html=True)
            
            if st.session_state.get('audio_data'):
                audio_file_path = st.session_state.audio_data
        
        else:
            # Fallback to file upload
            st.warning("Live recording not available. Please upload an audio file.")
            audio_file = st.file_uploader("Upload Audio", type=['wav', 'mp3', 'ogg'])
            if audio_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(audio_file.read())
                    audio_file_path = tmp_file.name
                st.audio(audio_file)
        
        st.info("💡 **Tip**: Speak clearly about your feelings for 30-60 seconds for best analysis")
    
    # Tab 3: Live Video/Photo
    with tab3:
        st.subheader("📹 Live Photo Capture")
        
        photo_method = st.radio(
            "Choose input method:",
            ["📸 Take Photo with Camera", "📁 Upload Photo"]
        )
        
        if "📸" in photo_method:
            # Live camera capture
            photo = st.camera_input("Take a photo")
            if photo:
                image = Image.open(photo)
                video_frame = np.array(image)
                st.success("✅ Photo captured!")
        else:
            # File upload
            uploaded_image = st.file_uploader("Upload Photo", type=['jpg', 'jpeg', 'png'])
            if uploaded_image:
                image = Image.open(uploaded_image)
                video_frame = np.array(image)
                st.image(image, caption="Uploaded Image", width=300)
    
    # Analysis button
    st.divider()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "🔍 Analyze My Mental State",
            type="primary",
            use_container_width=True
        )
    
    # Perform analysis
    if analyze_button:
        if not any([text_input, audio_file_path, video_frame is not None]):
            st.warning("⚠️ Please provide at least one input for analysis.")
            return
        
        with st.spinner("🧠 Analyzing your mental state..."):
            try:
                progress_bar = st.progress(0)
                
                progress_bar.progress(25)
                results = st.session_state.diagnosis_engine.comprehensive_analysis(
                    text=text_input if text_input and len(text_input.strip()) > 0 else None,
                    audio_file=audio_file_path,
                    video_frame=video_frame
                )
                
                progress_bar.progress(100)
                
                st.session_state.analysis_results = results
                st.session_state.current_mood = results['current_mood']['current_mood']
                st.session_state.conversation_manager.add_analysis_result(results)
                
                progress_bar.empty()
                st.success("✅ Analysis completed!")
                
                # Clean up temporary files
                if audio_file_path and os.path.exists(audio_file_path):
                    os.unlink(audio_file_path)
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")

def display_analysis_results():
    """Display analysis results"""
    if not st.session_state.analysis_results:
        return
    
    results = st.session_state.analysis_results
    
    if not results.get('analysis_successful', False):
        st.error("❌ Analysis was not successful. Please try again.")
        return
    
    st.header("📊 Analysis Results")
    
    # Primary Concern Highlight
    summary = results['summary']
    primary_concern = summary['primary_concern']
    confidence = summary['confidence']
    
    if primary_concern != 'None detected' and confidence > 10:
        st.markdown(f"""
        <div class="primary-concern-highlight">
            <h3>🎯 Primary Concern Detected: {primary_concern.title()}</h3>
            <p><strong>Confidence: {confidence}%</strong></p>
            <p><strong>Recommendation:</strong> Consider discussing this with a mental health professional.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Crisis Detection
    risk_level = summary['risk_level']
    if confidence > 50 and primary_concern in ['depression', 'self_harm'] and risk_level in ['moderate', 'high']:
        st.markdown(f'''<div class="crisis-alert">
        🚨 ELEVATED RISK DETECTED - Please seek immediate help:
        • National Suicide Prevention Lifeline: 988
        • Crisis Text Line: Text HOME to 741741
        • Emergency Services: 911
        </div>''', unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        concern_icon = "🔴" if confidence > 60 else "🟡" if confidence > 30 else "🟢"
        st.metric("🎯 PRIMARY CONCERN", f"{concern_icon} {primary_concern}", f"Confidence: {confidence}%")
    
    with col2:
        risk_colors = {'minimal': '🟢', 'low': '🟡', 'moderate': '🟠', 'high': '🔴'}
        risk_icon = risk_colors.get(risk_level, '⚪')
        st.metric("Risk Level", f"{risk_icon} {risk_level.title()}")
    
    with col3:
        mood_data = results['current_mood']
        mood = mood_data['current_mood']
        st.metric("Current Mood", f"😊 {mood}")
    
    with col4:
        needs_attention = summary['needs_attention']
        attention_text = "⚠️ Recommended" if needs_attention else "✅ Optional"
        st.metric("Professional Help", attention_text)

def display_chat_interface():
    """Chat interface"""
    st.header("💬 AI Therapist Support")
    
    messages = st.session_state.conversation_manager.get_current_messages()
    
    # Initialize conversation
    if not messages:
        if st.session_state.analysis_results:
            primary_concern = st.session_state.analysis_results.get('summary', {}).get('primary_concern', 'None detected')
            if primary_concern != 'None detected':
                initial_message = f"Hello! I noticed from your analysis that you might be experiencing some challenges with {primary_concern.lower()}. I'm here to listen and support you. How are you feeling about this right now?"
            else:
                initial_message = "Hello! I'm here to listen and support you. How are you feeling today?"
        else:
            initial_message = st.session_state.openai_client.generate_initial_conversation_starter()
        
        st.session_state.conversation_manager.add_message("assistant", initial_message)
        messages = st.session_state.conversation_manager.get_current_messages()
    
    # Display messages
    for message in messages:
        role = message['role']
        content = message['content']
        timestamp = message.get('timestamp', datetime.now())
        
        with st.chat_message(role, avatar="🤖" if role == "assistant" else "👤"):
            st.caption(f"**{timestamp.strftime('%H:%M')}**")
            st.markdown(content)
    
    # Crisis support notice
    st.info("🆘 **Crisis Support**: If you're having thoughts of self-harm, please call 988 (Suicide & Crisis Lifeline) or 911 immediately.")
    
    # Chat input
    user_input = st.chat_input("Share your thoughts...")
    
    if user_input:
        st.session_state.conversation_manager.add_message("user", user_input)
        
        is_crisis = st.session_state.openai_client.detect_crisis_keywords(user_input)
        
        with st.spinner("AI Therapist is responding..."):
            try:
                context = st.session_state.conversation_manager.get_conversation_context()
                
                if is_crisis:
                    ai_response = st.session_state.openai_client.generate_crisis_response(user_input)
                else:
                    ai_response = st.session_state.openai_client.generate_therapist_response(
                        user_input,
                        context,
                        st.session_state.analysis_results,
                        st.session_state.current_mood
                    )
                
                st.session_state.conversation_manager.add_message("assistant", ai_response)
                
            except Exception as e:
                error_msg = "I'm here to listen and support you. Could you tell me more about how you're feeling?"
                st.session_state.conversation_manager.add_message("assistant", error_msg)
                logger.error(f"Chat error: {e}")
        
        st.rerun()

def main():
    """Main application"""
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Mental Health Analyzer</h1>
        <p style="margin: 0;">AI-powered mental health assessment with live audio/video capture</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not initialize_app():
        return
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Session Info")
        
        if st.session_state.analysis_results:
            concern = st.session_state.analysis_results.get('summary', {}).get('primary_concern', 'None')
            risk = st.session_state.analysis_results.get('summary', {}).get('risk_level', 'minimal')
            st.write(f"**Primary Concern:** {concern}")
            st.write(f"**Risk Level:** {risk.title()}")
            st.write(f"**Current Mood:** {st.session_state.current_mood}")
        
        st.divider()
        
        if st.button("🔄 New Session"):
            st.session_state.analysis_results = None
            st.session_state.current_mood = "Unknown"
            st.success("New session started!")
            st.rerun()
    
    # Main content
    col1, col2 = st.columns([1.3, 0.7])
    
    with col1:
        display_analysis_section()
        
        if st.session_state.analysis_results:
            st.divider()
            display_analysis_results()
    
    with col2:
        display_chat_interface()
    
    # Footer
    st.markdown("---")
    st.markdown("**Disclaimer**: This tool is for informational purposes only. For mental health emergencies, please seek immediate professional help.")

if __name__ == "__main__":
    main()

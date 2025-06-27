import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import logging
import io

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

# Enhanced CSS (removed problematic chat styles)
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
    
    .audio-container {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
        background: #f8f9fa;
    }
    
    .mood-indicator {
        font-size: 2rem;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid;
    }
    
    .analysis-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
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
    
    .language-badge {
        background: #e8f5e8;
        color: #2e7d32;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .upload-section {
        border: 2px dashed #28a745;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        background: #f8fff9;
    }
    
    .instruction-box {
        background: #e7f3ff;
        border: 1px solid #b3d9ff;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Custom chat styling for better appearance */
    .stChatMessage {
        border-radius: 15px;
        margin: 0.5rem 0;
    }
    
    .stChatMessage[data-testid="chat-message-assistant"] {
        background-color: #f0f2f6;
    }
    
    .stChatMessage[data-testid="chat-message-user"] {
        background-color: #e3f2fd;
    }
</style>
""", unsafe_allow_html=True)

def initialize_app():
    """Initialize application components"""
    try:
        if 'diagnosis_engine' not in st.session_state:
            with st.spinner("🤖 Loading AI models... This may take a few moments."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Initializing text analysis models...")
                progress_bar.progress(25)
                
                status_text.text("Loading audio processing models...")
                progress_bar.progress(50)
                
                status_text.text("Setting up video analysis...")
                progress_bar.progress(75)
                
                st.session_state.diagnosis_engine = DiagnosisEngine(settings.openai_api_key)
                
                progress_bar.progress(100)
                status_text.text("All models loaded successfully! ✅")
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
        
        if 'conversation_manager' not in st.session_state:
            st.session_state.conversation_manager = ConversationManager()
        
        if 'openai_client' not in st.session_state:
            st.session_state.openai_client = OpenAIClient(settings.openai_api_key)
        
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        
        if 'current_mood' not in st.session_state:
            st.session_state.current_mood = "Unknown"
        
        if 'selected_language' not in st.session_state:
            st.session_state.selected_language = 'en'
        
        return True
        
    except Exception as e:
        st.error(f"❌ Failed to initialize application: {str(e)}")
        st.info("Please check your API keys and internet connection.")
        return False

def display_sidebar():
    """Enhanced sidebar"""
    with st.sidebar:
        st.header("🌐 Language Selection")
        
        language_options = [
            ("English", "en", "🇺🇸"),
            ("हिन्दी (Hindi)", "hi", "🇮🇳")
        ]
        
        selected_lang = st.selectbox(
            "Choose your language:",
            options=language_options,
            format_func=lambda x: f"{x[2]} {x[0]}",
            index=0 if st.session_state.get('selected_language', 'en') == 'en' else 1
        )
        
        st.session_state.selected_language = selected_lang[1]
        
        if selected_lang[1] == 'hi':
            st.markdown('<span class="language-badge">हिन्दी समर्थन सक्रिय</span>', unsafe_allow_html=True)
        
        st.divider()
        
        # Session statistics
        st.header("📊 Session Statistics")
        
        stats = st.session_state.conversation_manager.get_conversation_statistics()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Sessions", stats['total_conversations'])
            st.metric("Messages", stats['total_messages'])
        with col2:
            if stats['most_common_mood'] != 'None':
                st.metric("Common Mood", stats['most_common_mood'])
            else:
                st.metric("Status", "New User")
        
        if st.session_state.analysis_results:
            st.subheader("🎯 Current Session")
            current_concern = st.session_state.analysis_results.get('summary', {}).get('primary_concern', 'None')
            current_risk = st.session_state.analysis_results.get('summary', {}).get('risk_level', 'minimal')
            
            st.write(f"**Primary Concern:** {current_concern}")
            st.write(f"**Risk Level:** {current_risk.title()}")
            st.write(f"**Current Mood:** {st.session_state.current_mood}")
        
        st.divider()
        
        # Session controls
        st.header("⚙️ Session Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Session"):
                st.session_state.conversation_manager.save_current_conversation()
                st.success("Session saved!")
                time.sleep(1)
                st.rerun()
        
        with col2:
            if st.button("🔄 New Session"):
                st.session_state.conversation_manager.save_current_conversation()
                st.session_state.analysis_results = None
                st.success("New session started!")
                time.sleep(1)
                st.rerun()

def display_analysis_section():
    """Enhanced analysis section with working audio input"""
    language = st.session_state.get('selected_language', 'en')
    
    if language == 'hi':
        st.header("🎯 बहुआयामी मानसिक स्वास्थ्य विश्लेषण")
        st.write("टेक्स्ट, आवाज़ और फोटो के माध्यम से अपनी मानसिक स्थिति का विश्लेषण करें")
    else:
        st.header("🎯 Multi-Modal Mental Health Analysis")
        st.write("Analyze your mental state through text, voice, and photo inputs")
    
    # Input tabs
    if language == 'hi':
        tab_labels = ["💬 टेक्स्ट इनपुट", "🎤 ऑडियो अपलोड", "📹 फोटो कैप्चर"]
    else:
        tab_labels = ["💬 Text Input", "🎤 Audio Upload", "📹 Photo Capture"]
    
    tab1, tab2, tab3 = st.tabs(tab_labels)
    
    # Initialize input variables
    text_input = None
    audio_file_path = None
    video_frame = None
    
    # Tab 1: Text Input
    with tab1:
        if language == 'hi':
            st.subheader("अपने विचार साझा करें")
            placeholder_text = "आप कैसा महसूस कर रहे हैं? आपके मन में क्या बात है?\n\nउदाहरण:\n'मुझे बहुत चिंता हो रही है और रात में नींद नहीं आती...'\n'आज मैं बहुत खुश हूँ क्योंकि...'\n'मुझे लगता है कि मैं बहुत तनाव में हूँ...'"
        else:
            placeholder_text = "How are you feeling? What's on your mind?\n\nExamples:\n'I've been feeling anxious lately and can't sleep well...'\n'I'm really excited about my new job...'\n'I feel overwhelmed with everything going on...'"
        
        text_input = st.text_area(
            "Your thoughts:" if language == 'en' else "आपके विचार:",
            height=150,
            placeholder=placeholder_text,
            help="Express your current thoughts, feelings, or concerns in detail." if language == 'en' else "अपने वर्तमान विचार, भावनाएं या चिंताओं को विस्तार से व्यक्त करें।"
        )
        
        if text_input:
            word_count = len(text_input.split())
            st.caption(f"Word count: {word_count}" if language == 'en' else f"शब्द गिनती: {word_count}")
            
            # Auto-detect language
            if st.session_state.get('diagnosis_engine'):
                try:
                    detected_lang = st.session_state.diagnosis_engine.text_analyzer.detect_language(text_input)
                    if detected_lang == 'hi':
                        st.markdown('<span class="language-badge">हिन्दी भाषा पहचानी गई</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="language-badge">English detected</span>', unsafe_allow_html=True)
                except:
                    pass
    
    # Tab 2: Audio Upload (FIXED - No experimental features)
    with tab2:
        if language == 'hi':
            st.subheader("🎤 ऑडियो फ़ाइल अपलोड करें")
        else:
            st.subheader("🎤 Upload Audio File")
        
        st.markdown('<div class="audio-container">', unsafe_allow_html=True)
        
        # File upload option
        audio_file = st.file_uploader(
            "Choose an audio file" if language == 'en' else "ऑडियो फ़ाइल चुनें",
            type=settings.supported_audio_formats,
            help="Upload WAV, MP3, OGG, or M4A files" if language == 'en' else "WAV, MP3, OGG, या M4A फ़ाइलें अपलोड करें"
        )
        
        if audio_file:
            st.audio(audio_file, format='audio/wav')
            
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{audio_file.type.split("/")[1]}') as tmp_file:
                tmp_file.write(audio_file.read())
                audio_file_path = tmp_file.name
            
            if language == 'hi':
                st.success("✅ ऑडियो फ़ाइल अपलोड हुई!")
            else:
                st.success("✅ Audio file uploaded!")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Recording instructions
        st.markdown('<div class="instruction-box">', unsafe_allow_html=True)
        
        if language == 'hi':
            st.markdown("""
            ### 🎙️ ऑडियो कैसे रिकॉर्ड करें:
            
            **विकल्प 1: अपने फोन का उपयोग करें**
            1. अपने फोन पर वॉयस रिकॉर्डर ऐप खोलें
            2. 30-60 सेकंड तक अपनी भावनाओं के बारे में बात करें
            3. फ़ाइल को सेव करें और ऊपर अपलोड करें
            
            **विकल्प 2: कंप्यूटर का उपयोग करें**
            1. Windows Voice Recorder या Audacity का उपयोग करें
            2. 30-60 सेकंड तक स्पष्ट रूप से बोलें
            3. WAV या MP3 फॉर्मेट में सेव करें और अपलोड करें
            """)
        else:
            st.markdown("""
            ### 🎙️ How to Record Audio:
            
            **Option 1: Use your phone**
            1. Open voice recorder app on your phone
            2. Record 30-60 seconds describing your feelings
            3. Save the file and upload it above
            
            **Option 2: Use your computer**
            1. Use Windows Voice Recorder or Audacity
            2. Record yourself speaking clearly for 30-60 seconds
            3. Save as WAV or MP3 and upload above
            
            **Option 3: Online tools**
            1. Search "online voice recorder" in your browser
            2. Record and download the audio file
            3. Upload the file above
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Tips for better audio
        if language == 'hi':
            st.info("💡 **सुझाव**: शांत जगह में रिकॉर्ड करें। माइक के करीब और स्पष्ट रूप से बोलें। अपनी भावनाओं के बारे में खुलकर बात करें।")
        else:
            st.info("💡 **Tips**: Record in a quiet environment. Speak clearly and close to the microphone. Express your feelings openly for better analysis.")
    
    # Tab 3: Photo/Video Capture
    with tab3:
        if language == 'hi':
            st.subheader("📹 फोटो कैप्चर या अपलोड")
        else:
            st.subheader("📹 Photo Capture or Upload")
        
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        
        # Method selection
        photo_method = st.radio(
            "Choose photo input method:" if language == 'en' else "फोटो इनपुट विधि चुनें:",
            ["📸 Take Photo with Camera", "📁 Upload Image File"] if language == 'en' else ["📸 कैमरे से फोटो लें", "📁 इमेज फ़ाइल अपलोड करें"]
        )
        
        if "📸" in photo_method:
            # Built-in camera
            photo = st.camera_input(
                "Take a photo of yourself" if language == 'en' else "अपनी फोटो लें",
                help="Make sure your face is clearly visible and well-lit" if language == 'en' else "सुनिश्चित करें कि आपका चेहरा स्पष्ट रूप से दिखाई दे रहा है"
            )
            
            if photo:
                image = Image.open(photo)
                video_frame = np.array(image)
                
                if language == 'hi':
                    st.success("✅ फोटो सफलतापूर्वक ली गई!")
                else:
                    st.success("✅ Photo captured successfully!")
        
        else:
            # File upload
            uploaded_image = st.file_uploader(
                "Upload a clear photo of yourself" if language == 'en' else "अपनी स्पष्ट फोटो अपलोड करें",
                type=settings.supported_image_formats,
                help="Upload JPG, PNG, or other image formats" if language == 'en' else "JPG, PNG या अन्य इमेज फॉर्मेट अपलोड करें"
            )
            
            if uploaded_image:
                image = Image.open(uploaded_image)
                video_frame = np.array(image)
                st.image(image, caption="Uploaded Image" if language == 'en' else "अपलोड की गई इमेज", width=300)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Tips for better photos
        if language == 'hi':
            st.info("💡 **सुझाव**: अच्छी रोशनी में फोटो लें। कैमरा आपके चेहरे के सामने हो। प्राकृतिक भाव बनाए रखें।")
        else:
            st.info("💡 **Tips**: Take photo in good lighting. Face the camera directly. Keep a natural expression.")
    
    # Analysis button
    st.divider()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        button_text = "🔍 Analyze My Mental State" if language == 'en' else "🔍 मेरी मानसिक स्थिति का विश्लेषण करें"
        
        analyze_button = st.button(
            button_text,
            type="primary",
            use_container_width=True,
            help="Process all provided inputs and generate mental health insights" if language == 'en' else "सभी इनपुट को प्रोसेस करें और मानसिक स्वास्थ्य की जानकारी प्राप्त करें"
        )
    
    # Perform analysis
    if analyze_button:
        if not any([text_input, audio_file_path, video_frame is not None]):
            warning_text = "⚠️ Please provide at least one input (text, audio, or photo) for analysis." if language == 'en' else "⚠️ कृपया विश्लेषण के लिए कम से कम एक इनपुट (टेक्स्ट, ऑडियो या फोटो) प्रदान करें।"
            st.warning(warning_text)
            return
        
        spinner_text = "🧠 Analyzing your mental state... Please wait." if language == 'en' else "🧠 आपकी मानसिक स्थिति का विश्लेषण हो रहा है... कृपया प्रतीक्षा करें।"
        
        with st.spinner(spinner_text):
            try:
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Processing inputs..." if language == 'en' else "इनपुट प्रोसेस हो रहे हैं...")
                progress_bar.progress(25)
                
                # Perform comprehensive analysis
                results = st.session_state.diagnosis_engine.comprehensive_analysis(
                    text=text_input if text_input and len(text_input.strip()) > 0 else None,
                    audio_file=audio_file_path,
                    video_frame=video_frame
                )
                
                progress_bar.progress(75)
                status_text.text("Generating insights..." if language == 'en' else "सुझाव तैयार किए जा रहे हैं...")
                
                # Store results
                st.session_state.analysis_results = results
                st.session_state.current_mood = results['current_mood']['current_mood']
                
                # Add to conversation history
                st.session_state.conversation_manager.add_analysis_result(results)
                
                progress_bar.progress(100)
                status_text.text("Analysis complete! ✅" if language == 'en' else "विश्लेषण पूरा! ✅")
                
                # Clean up temporary files
                if audio_file_path and os.path.exists(audio_file_path):
                    os.unlink(audio_file_path)
                
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
                success_text = "✅ Analysis completed successfully!" if language == 'en' else "✅ विश्लेषण सफलतापूर्वक पूरा हुआ!"
                st.success(success_text)
                
            except Exception as e:
                error_text = f"❌ Analysis failed: {str(e)}" if language == 'en' else f"❌ विश्लेषण असफल: {str(e)}"
                st.error(error_text)
                logger.error(f"Analysis error: {str(e)}")
                return

def display_analysis_results():
    """Enhanced results display with primary concern focus"""
    if not st.session_state.analysis_results:
        return
    
    results = st.session_state.analysis_results
    language = st.session_state.get('selected_language', 'en')
    
    if not results.get('analysis_successful', False):
        error_text = "❌ Analysis was not successful. Please try again." if language == 'en' else "❌ विश्लेषण सफल नहीं हुआ। कृपया पुनः प्रयास करें।"
        st.error(error_text)
        return
    
    header_text = "📊 Analysis Results" if language == 'en' else "📊 विश्लेषण परिणाम"
    st.header(header_text)
    
    # Primary Concern Highlight (Enhanced Focus)
    summary = results['summary']
    primary_concern = summary['primary_concern']
    confidence = summary['confidence']
    
    if primary_concern != 'None detected' and confidence > 20:
        if language == 'hi':
            concern_text = f"🎯 मुख्य चिंता का पता चला: {primary_concern.title()}"
            confidence_text = f"विश्वसनीयता: {confidence}%"
            recommendation_text = "सिफारिश: यह आपकी मुख्य चिंता का क्षेत्र है। किसी मानसिक स्वास्थ्य पेशेवर से इस पर चर्चा करने पर विचार करें।"
        else:
            concern_text = f"🎯 Primary Concern Detected: {primary_concern.title()}"
            confidence_text = f"Confidence: {confidence}%"
            recommendation_text = "Recommendation: This appears to be your main area of concern. Consider discussing this with a mental health professional."
        
        st.markdown(f"""
        <div class="primary-concern-highlight">
            <h3>{concern_text}</h3>
            <p><strong>{confidence_text}</strong></p>
            <p><strong>{recommendation_text}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced Crisis Detection
    risk_level = summary['risk_level']
    if confidence > 60 and primary_concern in ['depression', 'anxiety', 'ptsd'] and risk_level in ['moderate', 'high']:
        crisis_text = """
        🚨 ELEVATED RISK DETECTED - Please consider seeking support:
        • Talk to a trusted friend, family member, or counselor
        • National Suicide Prevention Lifeline: 988
        • Crisis Text Line: Text HOME to 741741
        • For emergencies: 911
        """ if language == 'en' else """
        🚨 उच्च जोखिम का पता चला - कृपया सहायता लेने पर विचार करें:
        • किसी विश्वसनीय मित्र, परिवारजन या काउंसलर से बात करें
        • आपातकालीन सहायता: 112 (भारत)
        • तुरंत किसी मानसिक स्वास्थ्य पेशेवर से संपर्क करें
        """
        
        st.markdown(f'<div class="crisis-alert">{crisis_text}</div>', unsafe_allow_html=True)
    
    # Current mood display (smaller emphasis)
    mood_data = results['current_mood']
    mood = mood_data['current_mood']
    mood_confidence = mood_data['confidence']
    
    mood_colors = {
        'Happy': '#4CAF50', 'Sad': '#2196F3', 'Angry': '#F44336',
        'Anxious': '#FF9800', 'Neutral': '#9E9E9E', 'Excited': '#E91E63',
        'Calm': '#00BCD4', 'Surprised': '#FFEB3B', 'Tired': '#795548'
    }
    
    mood_color = mood_colors.get(mood, '#9E9E9E')
    
    # Summary metrics with PRIMARY CONCERN EMPHASIS
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # PRIMARY CONCERN gets the biggest emphasis
        concern_icon = "🔴" if confidence > 60 else "🟡" if confidence > 30 else "🟢"
        label = "🎯 PRIMARY CONCERN" if language == 'en' else "🎯 मुख्य चिंता"
        
        st.metric(
            label,
            f"{concern_icon} {primary_concern}",
            delta=f"Confidence: {confidence}%",
            help="Most likely mental health area needing attention" if language == 'en' else "सबसे अधिक ध्यान देने वाला मानसिक स्वास्थ्य क्षेत्र"
        )
    
    with col2:
        # Risk level
        risk_colors = {'minimal': '🟢', 'low': '🟡', 'moderate': '🟠', 'high': '🔴'}
        risk_icon = risk_colors.get(risk_level, '⚪')
        
        label = "Risk Level" if language == 'en' else "जोखिम स्तर"
        st.metric(
            label,
            f"{risk_icon} {risk_level.title()}",
            help="Overall risk assessment" if language == 'en' else "समग्र जोखिम मूल्यांकन"
        )
    
    with col3:
        # Current mood (less emphasis)
        label = "Current Mood" if language == 'en' else "वर्तमान मूड"
        st.metric(
            label,
            f"😊 {mood}",
            delta=f"{mood_confidence:.1%}",
            help="Current emotional state" if language == 'en' else "वर्तमान भावनात्मक स्थिति"
        )
    
    with col4:
        # Professional help needed
        needs_attention = summary['needs_attention']
        attention_text = "⚠️ Recommended" if needs_attention else "✅ Optional"
        if language == 'hi':
            attention_text = "⚠️ सुझाई गई" if needs_attention else "✅ वैकल्पिक"
        
        label = "Professional Help" if language == 'en' else "पेशेवर सहायता"
        st.metric(
            label,
            attention_text,
            help="Whether professional consultation is recommended" if language == 'en' else "क्या पेशेवर सलाह सुझाई जाती है"
        )
    
    # Detailed diagnosis with PRIMARY CONCERN FOCUS
    diagnosis = results['diagnosis']
    
    if diagnosis['top_conditions']:
        header = "🏥 Detailed Assessment - Primary Concerns First" if language == 'en' else "🏥 विस्तृत मूल्यांकन - मुख्य चिंताएं पहले"
        st.subheader(header)
        
        sorted_conditions = sorted(diagnosis['top_conditions'], key=lambda x: x['confidence_percentage'], reverse=True)
        
        for i, condition in enumerate(sorted_conditions):
            is_primary = (i == 0 and condition['confidence_percentage'] > 20)
            
            # Enhanced labeling for primary concern
            if is_primary:
                prefix = "🎯 PRIMARY CONCERN: " if language == 'en' else "🎯 मुख्य चिंता: "
                expansion = True
            else:
                prefix = f"#{i+1}: "
                expansion = False
            
            with st.expander(
                f"{prefix}{condition['condition'].title()} - {condition['confidence_percentage']}%",
                expanded=expansion
            ):
                severity_label = "Severity:" if language == 'en' else "गंभीरता:"
                description_label = "Description:" if language == 'en' else "विवरण:"
                confidence_label = "Assessment Confidence:" if language == 'en' else "मूल्यांकन विश्वसनीयता:"
                
                st.write(f"**{severity_label}** {condition['severity'].title()}")
                st.write(f"**{description_label}** {condition['description']}")
                st.write(f"**{confidence_label}** {condition['confidence_percentage']}%")
                
                # Enhanced progress bar for primary concern
                if is_primary:
                    st.markdown(f"""
                    <div style="background: linear-gradient(90deg, #dc3545 0%, #fd7e14 100%); 
                                height: 25px; width: {condition['confidence_percentage']}%; 
                                border-radius: 12px; display: flex; align-items: center; 
                                justify-content: center; color: white; font-weight: bold; margin: 10px 0;">
                        PRIMARY: {condition['confidence_percentage']}%
                    </div>
                    """, unsafe_allow_html=True)
                    
                    primary_info = "💡 This is your PRIMARY mental health concern based on comprehensive analysis." if language == 'en' else "💡 व्यापक विश्लेषण के आधार पर यह आपकी मुख्य मानसिक स्वास्थ्य चिंता है।"
                    st.info(primary_info)
                else:
                    # Standard progress bar for secondary concerns
                    st.progress(condition['confidence_percentage'] / 100)
    
    # Enhanced AI Analysis
    if diagnosis.get('ai_analysis'):
        header = "🤖 AI Therapist Professional Assessment" if language == 'en' else "🤖 AI थेरेपिस्ट पेशेवर मूल्यांकन"
        st.subheader(header)
        st.markdown(f"""
        <div class="analysis-card">
            {diagnosis['ai_analysis']}
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced Recommendations (Primary-concern focused)
    if diagnosis.get('recommendations'):
        header = "💡 Personalized Action Plan" if language == 'en' else "💡 व्यक्तिगत कार्य योजना"
        st.subheader(header)
        
        # Separate recommendations by priority
        primary_recs = []
        general_recs = []
        
        for rec in diagnosis['recommendations']:
            if any(keyword in rec.lower() for keyword in [primary_concern.lower(), 'professional', 'immediate']):
                primary_recs.append(rec)
            else:
                general_recs.append(rec)
        
        if primary_recs:
            priority_header = f"**🎯 Priority Actions for {primary_concern.title()}:**" if language == 'en' else f"**🎯 {primary_concern.title()} के लिए प्राथमिकता कार्य:**"
            st.markdown(priority_header)
            for i, rec in enumerate(primary_recs, 1):
                st.markdown(f"**{i}.** {rec}")
            
            st.divider()
        
        if general_recs:
            general_header = "**📋 Additional Wellness Recommendations:**" if language == 'en' else "**📋 अतिरिक्त कल्याण सुझाव:**"
            st.markdown(general_header)
            for i, rec in enumerate(general_recs, len(primary_recs) + 1):
                st.markdown(f"{i}. {rec}")
    
    # Language detection info
    text_analysis = results.get('individual_analyses', {}).get('text_analysis', {})
    detected_language = text_analysis.get('language', 'unknown')
    
    if detected_language == 'hi':
        st.info("🇮🇳 **हिन्दी भाषा का पता चला**: विश्लेषण हिन्दी भाषा के पैटर्न और सांस्कृतिक संदर्भ के लिए अनुकूलित किया गया है।")
    elif detected_language == 'en' and language == 'hi':
        st.info("🇺🇸 **अंग्रेजी भाषा का पता चला**: विश्लेषण अंग्रेजी भाषा के पैटर्न के लिए अनुकूलित किया गया है।")

def display_chat_interface():
    """Enhanced chat interface with native Streamlit chat components (FIXED)"""
    language = st.session_state.get('selected_language', 'en')
    
    header = "💬 AI Therapist - Personalized Support" if language == 'en' else "💬 AI थेरेपिस्ट - व्यक्तिगत सहायता"
    st.header(header)
    
    # Get primary concern for context
    primary_concern = "general support"
    if st.session_state.analysis_results:
        primary_concern = st.session_state.analysis_results.get('summary', {}).get('primary_concern', 'general support')
    
    # Enhanced conversation starter
    messages = st.session_state.conversation_manager.get_current_messages()
    
    if not messages and st.session_state.analysis_results:
        if primary_concern != 'None detected':
            if language == 'hi':
                initial_message = f"नमस्ते! मैंने देखा है कि आपके विश्लेषण में {primary_concern.lower()} से संबंधित संकेत हैं। मैं यहाँ आपकी बात सुनने और इस विषय पर विशेष ध्यान देते हुए आपका समर्थन करने के लिए हूँ। आप इस बारे में कैसा महसूस कर रहे हैं?"
            else:
                initial_message = f"Hello! I noticed from your analysis that there are indicators related to {primary_concern.lower()}. I'm here to listen and provide targeted support for this area. How are you feeling about this right now?"
        else:
            try:
                initial_message = st.session_state.openai_client.generate_initial_conversation_starter(
                    st.session_state.analysis_results
                )
            except:
                initial_message = "Hello! I'm here to listen and support you. How are you feeling today?" if language == 'en' else "नमस्ते! मैं यहाँ आपकी बात सुनने और आपका समर्थन करने के लिए हूँ। आज आप कैसा महसूस कर रहे हैं?"
        
        st.session_state.conversation_manager.add_message("assistant", initial_message)
        messages = st.session_state.conversation_manager.get_current_messages()
    
    # Context indicator
    if primary_concern != 'general support' and primary_concern != 'None detected':
        context_text = f"🎯 **Focus Area**: {primary_concern.title()}" if language == 'en' else f"🎯 **फोकस क्षेत्र**: {primary_concern.title()}"
        st.info(context_text)
    
    # FIXED: Chat display with native Streamlit chat components
    chat_container = st.container()
    
    with chat_container:
        for message in messages:
            role = message['role']
            content = message['content']
            timestamp = message.get('timestamp', datetime.now())
            
            # Use Streamlit's native chat message display
            with st.chat_message(role, avatar="🤖" if role == "assistant" else "👤"):
                st.caption(f"**{timestamp.strftime('%H:%M')}**")
                st.markdown(content)
    
    # Crisis support notice
    if language == 'hi':
        crisis_text = "🆘 **आपातकालीन सहायता**: यदि आप आत्म-हानि के विचारों से गुजर रहे हैं, तो कृपया तुरंत 112 (आपातकाल) पर कॉल करें।"
    else:
        crisis_text = "🆘 **Crisis Support**: If you're having thoughts of self-harm, please call 112 or 911 immediately."
    
    st.info(crisis_text)
    
    # Enhanced chat input with context
    input_placeholder = "Share your thoughts about this..." if language == 'en' else "इस बारे में अपने विचार साझा करें..."
    if primary_concern != 'general support' and primary_concern != 'None detected':
        input_placeholder = f"How are you feeling about {primary_concern.lower()}?" if language == 'en' else f"{primary_concern.lower()} के बारे में आप कैसा महसूस कर रहे हैं?"
    
    user_input = st.chat_input(input_placeholder, key="chat_input")
    
    if user_input:
        # Add user message
        st.session_state.conversation_manager.add_message("user", user_input)
        
        # Enhanced crisis detection
        is_crisis = st.session_state.openai_client.detect_crisis_keywords(user_input)
        
        # Generate contextual response
        spinner_text = "AI Therapist is preparing a personalized response..." if language == 'en' else "AI थेरेपिस्ट व्यक्तिगत प्रतिक्रिया तैयार कर रहा है..."
        
        with st.spinner(spinner_text):
            try:
                context = st.session_state.conversation_manager.get_conversation_context()
                
                if is_crisis:
                    ai_response = st.session_state.openai_client.generate_crisis_response(user_input)
                else:
                    ai_response = st.session_state.openai_client.generate_therapist_response(
                        user_input,
                        context,
                        st.session_state.analysis_results,
                        st.session_state.current_mood,
                        language=language
                    )
                
                st.session_state.conversation_manager.add_message("assistant", ai_response)
                
            except Exception as e:
                if language == 'hi':
                    error_msg = "मुझे खेद है, मुझे अभी जवाब देने में परेशानी हो रही है। कृपया पुनः प्रयास करें।"
                else:
                    error_msg = "I'm sorry, I'm having trouble responding right now. Please try again."
                
                st.session_state.conversation_manager.add_message("assistant", error_msg)
                logger.error(f"Chat response error: {str(e)}")
        
        st.rerun()

def main():
    """Main application with enhanced focus on primary concerns"""
    # Header
    language = st.session_state.get('selected_language', 'en')
    
    if language == 'hi':
        header_text = "🧠 मानसिक स्वास्थ्य विश्लेषक"
        subtitle_text = "AI-संचालित मानसिक स्वास्थ्य मूल्यांकन - मुख्य चिंताओं पर केंद्रित"
    else:
        header_text = "🧠 Mental Health Analyzer"
        subtitle_text = "AI-powered mental health assessment - Focused on Primary Concerns"
    
    st.markdown(f"""
    <div class="main-header">
        <h1>{header_text}</h1>
        <p style="margin: 0;">{subtitle_text}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize app
    if not initialize_app():
        return
    
    # Display sidebar
    display_sidebar()
    
    # Main content with enhanced layout
    col1, col2 = st.columns([1.3, 0.7])  # Adjusted ratio for better primary concern display
    
    with col1:
        # Analysis section
        display_analysis_section()
        
        # Results section with PRIMARY CONCERN EMPHASIS
        if st.session_state.analysis_results:
            st.divider()
            display_analysis_results()
    
    with col2:
        # Chat interface with primary concern context
        display_chat_interface()
    
    # Enhanced footer
    st.markdown("---")
    
    if language == 'hi':
        disclaimer_text = """
        **महत्वपूर्ण अस्वीकरण**: यह उपकरण केवल सूचनात्मक उद्देश्यों के लिए है। यह पेशेवर चिकित्सा सलाह, निदान या उपचार का विकल्प नहीं है। 
        मानसिक स्वास्थ्य संकट की स्थिति में तुरंत पेशेवर सहायता लें।
        """
    else:
        disclaimer_text = """
        **Important Disclaimer**: This tool is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. 
        For mental health emergencies, please seek immediate professional help.
        """
    
    st.markdown(disclaimer_text)
    
    # Debug info (only in development)
    if st.checkbox("🔧 Show Debug Info", value=False):
        st.subheader("Debug Information")
        st.write("**Session State Keys:**", list(st.session_state.keys()))
        st.write("**Selected Language:**", st.session_state.get('selected_language', 'en'))
        
        if st.session_state.analysis_results:
            with st.expander("Analysis Results JSON", expanded=False):
                st.json(st.session_state.analysis_results)

if __name__ == "__main__":
    main()

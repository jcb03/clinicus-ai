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

# Custom CSS for better styling
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
    
    .chat-container {
        height: 400px;
        overflow-y: auto;
        padding: 10px;
        background: #f8f9fa;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    
    .user-message {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0 5px 20px;
    }
    
    .assistant-message {
        background-color: #f3e5f5;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 20px 5px 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_app():
    """Initialize application components"""
    try:
        # Initialize components
        if 'diagnosis_engine' not in st.session_state:
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
        st.error(f"Failed to initialize application: {str(e)}")
        st.info("Please check your OpenAI API key and try refreshing the page.")
        return False

def display_sidebar():
    """Display sidebar with conversation history and statistics"""
    with st.sidebar:
        st.header("📝 Conversation History")
        
        # Statistics overview
        stats = st.session_state.conversation_manager.get_conversation_statistics()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Chats", stats['total_conversations'])
        with col2:
            st.metric("Total Messages", stats['total_messages'])
        
        if stats['most_common_mood'] != 'None':
            st.metric("Common Mood", stats['most_common_mood'])
        
        st.divider()
        
        # Conversation history
        conversations = st.session_state.conversation_manager.get_all_conversations()
        
        if conversations:
            st.subheader("Recent Sessions")
            for i, conv in enumerate(reversed(conversations[-5:])):  # Show last 5
                formatted_conv = st.session_state.conversation_manager.format_conversation_for_display(conv)
                
                with st.expander(f"💬 {formatted_conv['title']}", expanded=False):
                    st.write(f"**Duration:** {formatted_conv['duration']}")
                    st.write(f"**Messages:** {formatted_conv['messages']}")
                    st.write(f"**Primary Mood:** {formatted_conv['primary_mood']}")
                    st.write(f"**Risk Level:** {formatted_conv['risk_level']}")
                    st.write(f"**Preview:** {formatted_conv['preview']}")
                    
                    if st.button(f"Delete {formatted_conv['title']}", key=f"delete_{i}"):
                        st.session_state.conversation_manager.delete_conversation(formatted_conv['id'])
                        st.rerun()
        else:
            st.info("No conversations yet. Start by analyzing your mental state!")
        
        st.divider()
        
        # Session controls
        st.subheader("Session Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Session"):
                st.session_state.conversation_manager.save_current_conversation()
                st.success("Session saved!")
                time.sleep(1)
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear All"):
                st.session_state.conversation_manager.clear_all_conversations()
                st.success("All conversations cleared!")
                time.sleep(1)
                st.rerun()

def display_analysis_section():
    """Display multi-modal analysis section"""
    st.header("🎯 Multi-Modal Mental Health Analysis")
    st.write("Analyze your mental state through text, voice, and video inputs")
    
    # Input tabs
    tab1, tab2, tab3 = st.tabs(["💬 Text Input", "🎤 Audio Input", "📹 Video Input"])
    
    # Initialize input variables
    text_input = None
    audio_file_path = None
    video_frame = None
    
    with tab1:
        st.subheader("Share Your Thoughts")
        text_input = st.text_area(
            "How are you feeling? What's on your mind?",
            height=150,
            placeholder="I've been feeling anxious lately... / I'm really excited about... / I can't seem to concentrate...",
            help="Express your current thoughts, feelings, or concerns. The more detail you provide, the better the analysis."
        )
        
        # Language selection
        language = st.selectbox(
            "Select Language",
            options=settings.supported_languages,
            index=0,
            help="Choose the language of your text input"
        )
    
    with tab2:
        st.subheader("Voice Analysis")
        
        # Audio file upload
        audio_file = st.file_uploader(
            "Upload Audio File",
            type=settings.supported_audio_formats,
            help="Upload a voice recording (WAV, MP3, OGG, M4A)"
        )
        
        if audio_file:
            st.audio(audio_file, format='audio/wav')
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{audio_file.type.split("/")[1]}') as tmp_file:
                tmp_file.write(audio_file.read())
                audio_file_path = tmp_file.name
        
        st.info("💡 Tip: Record a 30-60 second clip describing how you feel for best results")
    
    with tab3:
        st.subheader("Facial Expression Analysis")
        
        video_option = st.radio(
            "Choose Video Input Method",
            ["📸 Take Photo", "📁 Upload Image"],
            help="Select how you want to provide visual input for emotion analysis"
        )
        
        if video_option == "📸 Take Photo":
            photo = st.camera_input("Take a photo of yourself")
            if photo:
                image = Image.open(photo)
                video_frame = np.array(image)
                st.success("Photo captured successfully!")
        
        elif video_option == "📁 Upload Image":
            uploaded_image = st.file_uploader(
                "Upload an image",
                type=settings.supported_image_formats,
                help="Upload a clear photo showing your face"
            )
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
            use_container_width=True,
            help="Process all provided inputs and generate mental health insights"
        )
    
    # Perform analysis
    if analyze_button:
        if not any([text_input, audio_file_path, video_frame is not None]):
            st.warning("⚠️ Please provide at least one input (text, audio, or video) for analysis.")
            return
        
        with st.spinner("🧠 Analyzing your mental state... This may take a moment."):
            try:
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Perform comprehensive analysis
                status_text.text("Processing inputs...")
                progress_bar.progress(25)
                
                results = st.session_state.diagnosis_engine.comprehensive_analysis(
                    text=text_input if text_input and len(text_input.strip()) > 0 else None,
                    audio_file=audio_file_path,
                    video_frame=video_frame
                )
                
                progress_bar.progress(75)
                status_text.text("Generating insights...")
                
                # Store results
                st.session_state.analysis_results = results
                st.session_state.current_mood = results['current_mood']['current_mood']
                
                # Add to conversation history
                st.session_state.conversation_manager.add_analysis_result(results)
                
                progress_bar.progress(100)
                status_text.text("Analysis complete! ✅")
                
                # Clean up temporary files
                if audio_file_path and os.path.exists(audio_file_path):
                    os.unlink(audio_file_path)
                
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
                st.success("✅ Analysis completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                return

def display_analysis_results():
    """Display analysis results"""
    if not st.session_state.analysis_results:
        return
    
    results = st.session_state.analysis_results
    
    if not results.get('analysis_successful', False):
        st.error("❌ Analysis was not successful. Please try again.")
        return
    
    st.header("📊 Analysis Results")
    
    # Current mood display
    mood_data = results['current_mood']
    mood = mood_data['current_mood']
    mood_confidence = mood_data['confidence']
    
    # Mood indicator with color coding
    mood_colors = {
        'Happy': '#4CAF50', 'Sad': '#2196F3', 'Angry': '#F44336',
        'Anxious': '#FF9800', 'Neutral': '#9E9E9E', 'Excited': '#E91E63',
        'Calm': '#00BCD4', 'Surprised': '#FFEB3B', 'Tired': '#795548'
    }
    
    mood_color = mood_colors.get(mood, '#9E9E9E')
    
    st.markdown(f"""
    <div class="mood-indicator" style="background-color: {mood_color}20; border-color: {mood_color};">
        <h2 style="color: {mood_color}; margin: 0;">Current Mood: {mood}</h2>
        <p style="margin: 0.5rem 0 0 0;">Confidence: {mood_confidence:.1%}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Summary metrics
    summary = results['summary']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Primary Concern",
            summary['primary_concern'],
            help="The most likely mental health area that may need attention"
        )
    
    with col2:
        confidence_pct = summary['confidence']
        st.metric(
            "Confidence",
            f"{confidence_pct}%",
            help="How confident the analysis is in the primary concern"
        )
    
    with col3:
        risk_level = summary['risk_level']
        risk_colors = {'minimal': '🟢', 'low': '🟡', 'moderate': '🟠', 'high': '🔴'}
        risk_icon = risk_colors.get(risk_level, '⚪')
        st.metric(
            "Risk Level",
            f"{risk_icon} {risk_level.title()}",
            help="Overall risk assessment based on all inputs"
        )
    
    with col4:
        needs_attention = "⚠️ Yes" if summary['needs_attention'] else "✅ No"
        st.metric(
            "Needs Attention",
            needs_attention,
            help="Whether professional consultation is recommended"
        )
    
    # Detailed diagnosis
    diagnosis = results['diagnosis']
    
    if diagnosis['top_conditions']:
        st.subheader("🏥 Detailed Assessment")
        
        for i, condition in enumerate(diagnosis['top_conditions']):
            with st.expander(f"#{i+1}: {condition['condition'].title()} - {condition['confidence_percentage']}%", expanded=i==0):
                st.write(f"**Severity:** {condition['severity'].title()}")
                st.write(f"**Description:** {condition['description']}")
                st.write(f"**Confidence:** {condition['confidence_percentage']}%")
                
                # Progress bar for confidence
                st.progress(condition['score'])
    
    # AI Analysis
    if diagnosis.get('ai_analysis'):
        st.subheader("🤖 AI Therapist Insights")
        st.markdown(f"""
        <div class="analysis-card">
            {diagnosis['ai_analysis']}
        </div>
        """, unsafe_allow_html=True)
    
    # Recommendations
    if diagnosis.get('recommendations'):
        st.subheader("💡 Personalized Recommendations")
        
        for i, rec in enumerate(diagnosis['recommendations'], 1):
            st.write(f"{i}. {rec}")
    
    # Emotion breakdown
    emotion_analysis = diagnosis.get('emotion_analysis', {})
    if emotion_analysis.get('combined_emotions'):
        st.subheader("😊 Emotion Analysis")
        
        emotions_df = pd.DataFrame([
            {'Emotion': emotion.title(), 'Score': score}
            for emotion, score in emotion_analysis['combined_emotions'].items()
        ]).sort_values('Score', ascending=True)
        
        fig = px.bar(
            emotions_df, 
            x='Score', 
            y='Emotion',
            orientation='h',
            title="Detected Emotions",
            color='Score',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def display_chat_interface():
    """Display chat interface"""
    st.header("💬 AI Therapist Chat")
    
    # Initialize conversation if needed
    messages = st.session_state.conversation_manager.get_current_messages()
    
    # If no messages and we have analysis results, start conversation
    if not messages and st.session_state.analysis_results:
        initial_message = st.session_state.openai_client.generate_initial_conversation_starter(
            st.session_state.analysis_results
        )
        st.session_state.conversation_manager.add_message("assistant", initial_message)
        messages = st.session_state.conversation_manager.get_current_messages()
    
    # Chat container
    chat_html = '<div class="chat-container">'
    
    for message in messages:
        role = message['role']
        content = message['content']
        timestamp = message.get('timestamp', datetime.now())
        
        if role == 'user':
            chat_html += f'''
            <div class="user-message">
                <small style="opacity: 0.7;">{timestamp.strftime('%H:%M')}</small><br>
                <strong>You:</strong> {content}
            </div>
            '''
        else:  # assistant
            chat_html += f'''
            <div class="assistant-message">
                <small style="opacity: 0.7;">{timestamp.strftime('%H:%M')}</small><br>
                <strong>🤖 AI Therapist:</strong> {content}
            </div>
            '''
    
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)
    
    # Crisis support notice
    st.info("🆘 **Crisis Support:** If you're having thoughts of self-harm, please call 988 (Suicide & Crisis Lifeline) or 911 immediately.")
    
    # Chat input
    user_input = st.chat_input(
        "Type your message here...",
        key="chat_input"
    )
    
    if user_input:
        # Add user message
        st.session_state.conversation_manager.add_message("user", user_input)
        
        # Check for crisis keywords
        is_crisis = st.session_state.openai_client.detect_crisis_keywords(user_input)
        
        # Generate response
        with st.spinner("AI Therapist is thinking..."):
            try:
                if is_crisis:
                    ai_response = st.session_state.openai_client.generate_crisis_response(user_input)
                else:
                    context = st.session_state.conversation_manager.get_conversation_context()
                    ai_response = st.session_state.openai_client.generate_therapist_response(
                        user_input,
                        context,
                        st.session_state.analysis_results,
                        st.session_state.current_mood
                    )
                
                # Add AI message
                st.session_state.conversation_manager.add_message("assistant", ai_response)
                
            except Exception as e:
                error_message = f"I'm sorry, I'm having trouble responding right now. Please try again. ({str(e)})"
                st.session_state.conversation_manager.add_message("assistant", error_message)
        
        # Refresh the page to show new messages
        st.rerun()

def main():
    """Main application function"""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Mental Health Analyzer</h1>
        <p style="margin: 0;">AI-powered mental health assessment through text, voice, and facial analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize app
    if not initialize_app():
        return
    
    # Display sidebar
    display_sidebar()
    
    # Main content area
    col1, col2 = st.columns([1.2, 0.8])
    
    with col1:
        # Analysis section
        display_analysis_section()
        
        # Results section
        if st.session_state.analysis_results:
            st.divider()
            display_analysis_results()
    
    with col2:
        # Chat interface
        display_chat_interface()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Disclaimer:** This tool is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. "
        "If you're experiencing a mental health crisis, please contact emergency services or a mental health professional immediately."
    )

if __name__ == "__main__":
    main()

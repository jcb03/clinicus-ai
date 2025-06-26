# 🧠 ClinicusAI - AI-Powered Mental Health Analyzer

An AI-powered mental health assessment platform that analyzes text, voice, and facial expressions using MediaPipe to provide insights into mental well-being.

## Features

- **Multi-modal Analysis**: Text sentiment, voice emotion, and facial expression analysis
- **AI-Powered Insights**: OpenAI GPT integration for personalized assessments
- **Real-time Chat**: Therapeutic conversation with AI therapist
- **Mood Tracking**: Timeline visualization of emotional states
- **Privacy-Focused**: No database storage, conversation history in memory only
- **MediaPipe Integration**: Advanced facial analysis without compilation issues

## Installation

### Prerequisites
- Python 3.11
- OpenAI API key
- Webcam (optional, for video analysis)
- Microphone (optional, for audio recording)

### Setup with UV

1. **Clone and navigate to project:**

git clone https://github.com/jcb03/clinicus-ai
cd mental_health_analyzer

2. **Initialize UV project:**

uv init --python 3.11

3. **Install dependencies:**

Core dependencies
uv add "streamlit>=1.29.0"
uv add "fastapi>=0.104.1"
uv add "uvicorn>=0.24.0"

AI/ML dependencies
uv add "transformers>=4.35.0"
uv add "torch>=2.1.0"
uv add "openai>=1.3.0"

Computer vision (MediaPipe instead of face-recognition)
uv add "opencv-python>=4.8.1.78"
uv add "mediapipe>=0.10.0"
uv add "Pillow>=10.0.1"

Audio processing
uv add "librosa>=0.10.1"
uv add "soundfile>=0.12.1"

Scientific computing
uv add "numpy>=1.24.3"
uv add "pandas>=2.0.3"
uv add "scipy>=1.11.1"
uv add "scikit-learn>=1.3.0"

Visualization
uv add "plotly>=5.17.0"
uv add "matplotlib>=3.7.1"
uv add "seaborn>=0.12.2"

Additional dependencies
uv add "streamlit-webrtc>=0.47.1"
uv add "websockets>=12.0"
uv add "python-multipart>=0.0.6"
uv add "pydantic>=2.5.0"
uv add "pydantic-settings>=2.0.3"
uv add "python-dotenv>=1.0.0"

5. **Set up environment variables:**
Create a `.env` file:

OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here

6. **Activate virtual environment:**

On Windows
.venv\Scripts\activate

On macOS/Linux
source .venv/bin/activate

7. **Run the application:**

Using UV
uv run streamlit run app/main.py

Or traditional method
python scripts/run_streamlit.py


### Setup with Traditional PIP

1. **Create virtual environment:**

python -m venv venv
venv\Scripts\activate # Windows

source venv/bin/activate # macOS/Linux


2. **Install dependencies:**

pip install -r requirements.txt


3. **Run application:**

streamlit run app/main.py

## Usage

1. **Text Analysis**: Enter your thoughts and feelings in the text area
2. **Audio Analysis**: Upload an audio file or record your voice  
3. **Video Analysis**: Take a photo or upload an image
4. **Get Insights**: Click "Analyze My Mental State" to process your inputs
5. **Chat**: Engage with the AI therapist for support and guidance

## Project Structure

mental_health_analyzer/
├── .env # Environment variables
├── .gitignore # Git ignore file
├── README.md # This file
├── pyproject.toml # UV/Python project configuration
├── requirements.txt # PIP requirements
├── Procfile # Render deployment
├── render.yaml # Render configuration
├── app/ # Main application
│ ├── init.py
│ ├── main.py # Streamlit main app
│ ├── config/ # Configuration
│ │ ├── init.py
│ │ └── settings.py
│ ├── models/ # Analysis models
│ │ ├── init.py
│ │ ├── text_analyzer.py
│ │ ├── audio_analyzer.py
│ │ ├── video_analyzer.py
│ │ └── diagnosis_engine.py
│ ├── utils/ # Utility functions
│ │ ├── init.py
│ │ ├── conversation_manager.py
│ │ └── openai_client.py
│ └── api/ # FastAPI backend
│ ├── init.py
│ └── fastapi_backend.py
├── scripts/ # Run scripts
│ ├── run_streamlit.py
│ └── run_fastapi.py
├── data/ # Data directory
├── logs/ # Log files
└── temp/ # Temporary files


## Models Used

- **Text Analysis**: 
  - Sentiment: `cardiffnlp/twitter-roberta-base-sentiment-latest`
  - Emotion: `j-hartmann/emotion-english-distilroberta-base`

- **Audio Analysis**:
  - Speech-to-Text: `openai/whisper-base`
  - Emotion: `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`

- **Video Analysis**:
  - Face Detection & Analysis: **MediaPipe** (Google's ML framework)
  - Facial Expression: `trpakov/vit-face-expression` (fallback)

## Why MediaPipe?

We use **MediaPipe** instead of `face-recognition` because:
- ✅ No CMake compilation required on Windows
- ✅ Better performance and accuracy
- ✅ Real-time processing capabilities
- ✅ Comprehensive facial landmark detection
- ✅ Maintained by Google with regular updates

## API Endpoints (FastAPI)

- `GET /health` - Health check
- `POST /analyze/text` - Analyze text input
- `POST /analyze/audio` - Analyze audio file
- `POST /analyze/combined` - Multi-modal analysis

Run FastAPI server:

python scripts/run_fastapi.py

API available at http://localhost:8000


## Deployment

### Render Deployment

1. Push your code to GitHub
2. Connect your GitHub repo to Render
3. Set environment variables in Render dashboard:
   - `OPENAI_API_KEY`
   - `HUGGINGFACE_TOKEN`
4. Deploy using the provided `render.yaml` configuration

### Local Development

Run Streamlit app
python scripts/run_streamlit.py

Available at http://localhost:8501
Run FastAPI backend (optional)
python scripts/run_fastapi.py

Available at http://localhost:8000


## Troubleshooting

### Common Issues

1. **CMake Error (if using face-recognition)**: 
   - Solution: We use MediaPipe instead, which doesn't require CMake

2. **OpenAI API Error**:
   - Check your API key is valid and has credits
   - Ensure `.env` file is in the root directory

3. **Model Loading Issues**:
   - Ensure you have internet connection for first-time model downloads
   - Models are cached locally after first download

4. **Audio/Video Upload Issues**:
   - Check file formats are supported
   - Ensure file size is under 50MB limit

### Performance Tips

- First run will be slower due to model downloads
- Use smaller audio files (30-60 seconds) for faster processing
- Ensure good lighting for facial analysis
- Clear speech in quiet environment for audio analysis

## Disclaimer

⚠️ **Important**: This application is for informational and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. If you are experiencing a mental health crisis, please contact:

- **Emergency Services**: 911
- **National Suicide Prevention Lifeline**: 988
- **Crisis Text Line**: Text HOME to 741741

Always consult with qualified mental health professionals for proper assessment and treatment.

## Privacy & Data

- No personal data is stored permanently
- Conversation history is kept in browser memory only
- Audio/video files are processed temporarily and deleted
- Analysis results are not transmitted to external servers (except OpenAI API)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section above

## Acknowledgments

- Hugging Face for pre-trained models
- OpenAI for GPT integration
- Google MediaPipe for facial analysis
- Streamlit for the web framework
- The open-source community for various libraries used

Setup Instructions Summary
To set up and run your complete mental health analyzer:

Create the project structure as shown above

Copy all the code files into their respective locations

Navigate to project directory:

The application will be available at http://localhost:8501 and will provide:

✅ Multi-modal mental health analysis (text, audio, video)

✅ AI-powered diagnosis and recommendations

✅ Real-time chat with AI therapist

✅ Mood tracking and visualization

✅ Conversation history management

✅ Crisis detection and response

✅ MediaPipe-based facial analysis (no compilation issues)



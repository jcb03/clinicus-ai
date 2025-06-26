from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import tempfile
import os
import logging
from datetime import datetime

from models.diagnosis_engine import DiagnosisEngine
from config.settings import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Mental Health Analyzer API",
    description="API for multi-modal mental health analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize diagnosis engine
try:
    diagnosis_engine = DiagnosisEngine(settings.openai_api_key)
    logger.info("Diagnosis engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize diagnosis engine: {str(e)}")
    diagnosis_engine = None

# Pydantic models
class TextAnalysisRequest(BaseModel):
    text: str
    language: Optional[str] = "en"

class AnalysisResponse(BaseModel):
    success: bool
    analysis_results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    version: str

# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

# Text analysis endpoint
@app.post("/analyze/text", response_model=AnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """Analyze text for mental health indicators"""
    try:
        if not diagnosis_engine:
            raise HTTPException(status_code=503, detail="Analysis engine not available")
        
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text input cannot be empty")
        
        # Perform text analysis
        results = diagnosis_engine.comprehensive_analysis(text=request.text)
        
        return AnalysisResponse(
            success=True,
            analysis_results=results,
            timestamp=datetime.now().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text analysis failed: {str(e)}")
        return AnalysisResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now().isoformat()
        )

# Audio analysis endpoint
@app.post("/analyze/audio", response_model=AnalysisResponse)
async def analyze_audio(audio_file: UploadFile = File(...)):
    """Analyze audio file for mental health indicators"""
    try:
        if not diagnosis_engine:
            raise HTTPException(status_code=503, detail="Analysis engine not available")
        
        # Validate file type
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            content = await audio_file.read()
            tmp_file.write(content)
            temp_path = tmp_file.name
        
        try:
            # Perform audio analysis
            results = diagnosis_engine.comprehensive_analysis(audio_file=temp_path)
            
            return AnalysisResponse(
                success=True,
                analysis_results=results,
                timestamp=datetime.now().isoformat()
            )
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio analysis failed: {str(e)}")
        return AnalysisResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now().isoformat()
        )

# Combined analysis endpoint
@app.post("/analyze/combined", response_model=AnalysisResponse)
async def analyze_combined(
    text: Optional[str] = Form(None),
    audio_file: Optional[UploadFile] = File(None),
    image_file: Optional[UploadFile] = File(None)
):
    """Perform combined analysis with multiple input types"""
    try:
        if not diagnosis_engine:
            raise HTTPException(status_code=503, detail="Analysis engine not available")
        
        # Validate at least one input
        if not any([text, audio_file, image_file]):
            raise HTTPException(status_code=400, detail="At least one input type is required")
        
        audio_path = None
        video_frame = None
        
        try:
            # Handle audio file
            if audio_file and audio_file.content_type.startswith('audio/'):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    content = await audio_file.read()
                    tmp_file.write(content)
                    audio_path = tmp_file.name
            
            # Handle image file
            if image_file and image_file.content_type.startswith('image/'):
                from PIL import Image
                import numpy as np
                import io
                
                content = await image_file.read()
                image = Image.open(io.BytesIO(content))
                video_frame = np.array(image)
            
            # Perform combined analysis
            results = diagnosis_engine.comprehensive_analysis(
                text=text if text and len(text.strip()) > 0 else None,
                audio_file=audio_path,
                video_frame=video_frame
            )
            
            return AnalysisResponse(
                success=True,
                analysis_results=results,
                timestamp=datetime.now().isoformat()
            )
        
        finally:
            # Clean up temporary files
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Combined analysis failed: {str(e)}")
        return AnalysisResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now().isoformat()
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

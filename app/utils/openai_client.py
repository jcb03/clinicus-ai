from openai import OpenAI
import logging
from typing import Dict, List, Optional
import time
from config.settings import settings

logger = logging.getLogger(__name__)

class OpenAIClient:
    def __init__(self, api_key: str = None):
        """Initialize OpenAI client"""
        self.api_key = api_key or settings.openai_api_key
        self.client = None
        
        if self.api_key and self.api_key != "your_openai_api_key_here":
            try:
                self.client = OpenAI(api_key=self.api_key)
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
        else:
            logger.warning("OpenAI API key not provided")
        
        self.default_model = settings.openai_model
        self.default_max_tokens = settings.max_tokens
        self.default_temperature = settings.temperature
        self.last_request_time = 0
        self.min_request_interval = 1
    
    def _rate_limit(self):
        """Rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def generate_therapist_response(self, 
                                 user_message: str,
                                 conversation_context: str = "",
                                 analysis_results: Optional[Dict] = None,
                                 mood: str = "neutral",
                                 language: str = "en") -> str:
        """Generate therapist response - FIXED"""
        
        # Check if client is available
        if not self.client:
            return "I'm here to listen and support you. Could you tell me more about how you're feeling? (Note: Enhanced AI responses require API configuration)"
        
        try:
            self._rate_limit()
            
            # Build system prompt
            system_prompt = """You are a compassionate, professional AI therapist assistant. Your role is to:

1. Provide empathetic, supportive responses
2. Ask thoughtful questions to help users explore their feelings
3. Offer gentle guidance and coping strategies
4. Always encourage professional help for serious concerns
5. Maintain appropriate boundaries
6. Never diagnose or provide medical advice

Guidelines:
- Be warm, understanding, and non-judgmental
- Use active listening techniques in your responses
- Validate the user's feelings
- Suggest healthy coping mechanisms when appropriate
- If the user expresses thoughts of self-harm, immediately encourage them to seek emergency help"""

            # Add context if available
            if analysis_results:
                primary_concern = analysis_results.get('summary', {}).get('primary_concern', 'None')
                risk_level = analysis_results.get('summary', {}).get('risk_level', 'minimal')
                
                if primary_concern != 'None detected':
                    system_prompt += f"\n\nIMPORTANT: The user's analysis suggests concerns related to {primary_concern} with {risk_level} risk level. Be extra supportive and consider gently encouraging professional consultation."
            
            # Build user prompt
            user_prompt = f"User's current message: {user_message}"
            
            if conversation_context:
                user_prompt = f"Recent conversation context:\n{conversation_context}\n\n{user_prompt}"
            
            user_prompt += "\n\nPlease respond as a supportive therapist, acknowledging what the user has shared and helping them explore their feelings further. Keep your response concise but meaningful."
            
            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.default_max_tokens,
                temperature=self.default_temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI request failed: {str(e)}")
            return "I'm here to listen and support you. Could you tell me more about how you're feeling? I want to understand what you're going through."
    
    def detect_crisis_keywords(self, text: str) -> bool:
        """Detect crisis keywords"""
        crisis_keywords = [
            'suicide', 'kill myself', 'end my life', 'want to die', 'better off dead',
            'no point living', 'end it all', 'hurt myself', 'self harm', 'cut myself'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in crisis_keywords)
    
    def generate_crisis_response(self, user_message: str, language: str = "en") -> str:
        """Generate crisis response"""
        return """I'm very concerned about you. Please reach out for immediate help:

🆘 IMMEDIATE HELP:
• National Suicide Prevention Lifeline: 988
• Crisis Text Line: Text HOME to 741741
• Emergency Services: 911 or 112 (India)
• For immediate danger, go to your nearest emergency room

You don't have to go through this alone. There are people who want to help you."""
    
    def generate_initial_conversation_starter(self, analysis_results: Optional[Dict] = None) -> str:
        """Generate conversation starter"""
        if analysis_results and analysis_results.get('summary', {}).get('primary_concern') != 'None detected':
            primary_concern = analysis_results['summary']['primary_concern']
            return f"Hello! I'm here to listen and support you. I noticed you might be dealing with some challenges related to {primary_concern}. How are you feeling right now?"
        else:
            return "Hello! I'm here to listen and support you. How are you feeling today? What's on your mind?"

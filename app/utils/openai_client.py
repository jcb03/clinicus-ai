import openai
import logging
from typing import Dict, List, Optional
import time
import json
from config.settings import settings

logger = logging.getLogger(__name__)

class OpenAIClient:
    def __init__(self, api_key: str = None):
        """Initialize OpenAI client"""
        self.api_key = api_key or settings.openai_api_key
        openai.api_key = self.api_key
        
        # Default parameters
        self.default_model = settings.openai_model
        self.default_max_tokens = settings.max_tokens
        self.default_temperature = settings.temperature
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1  # seconds
    
    def _rate_limit(self):
        """Simple rate limiting"""
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
                                 mood: str = "neutral") -> str:
        """Generate therapist response based on user input and context"""
        try:
            self._rate_limit()
            
            # Build context-aware prompt
            system_prompt = self._build_therapist_system_prompt(analysis_results, mood)
            user_prompt = self._build_therapist_user_prompt(user_message, conversation_context, analysis_results)
            
            response = openai.ChatCompletion.create(
                model=self.default_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.default_max_tokens,
                temperature=self.default_temperature,
                presence_penalty=0.3,
                frequency_penalty=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except openai.error.RateLimitError:
            logger.warning("OpenAI rate limit exceeded")
            return "I need a moment to process. Please try again shortly."
        
        except openai.error.AuthenticationError:
            logger.error("OpenAI authentication failed")
            return "I'm having trouble connecting right now. Please check your settings."
        
        except Exception as e:
            logger.error(f"OpenAI request failed: {str(e)}")
            return "I'm sorry, I'm having difficulty responding right now. How are you feeling, and is there anything specific you'd like to talk about?"
    
    def _build_therapist_system_prompt(self, analysis_results: Optional[Dict], mood: str) -> str:
        """Build system prompt for therapist responses"""
        base_prompt = """You are a compassionate, professional AI therapist assistant. Your role is to:

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

        # Add context-specific guidance
        if analysis_results:
            risk_level = analysis_results.get('summary', {}).get('risk_level', 'minimal')
            primary_concern = analysis_results.get('summary', {}).get('primary_concern', 'None')
            
            if risk_level in ['moderate', 'high']:
                base_prompt += f"\n\nIMPORTANT: The user's recent analysis suggests {risk_level} risk level with primary concern: {primary_concern}. Be extra supportive and gently encourage professional consultation."
            
            if primary_concern and primary_concern != 'None detected':
                base_prompt += f"\n\nThe user may be experiencing signs related to {primary_concern}. Tailor your response with appropriate sensitivity to this concern."
        
        if mood and mood.lower() != 'neutral':
            base_prompt += f"\n\nThe user's current mood appears to be: {mood}. Acknowledge this appropriately in your response."
        
        return base_prompt
    
    def _build_therapist_user_prompt(self, user_message: str, context: str, analysis_results: Optional[Dict]) -> str:
        """Build user prompt with context"""
        prompt_parts = []
        
        if context:
            prompt_parts.append(f"Recent conversation context:\n{context}\n")
        
        if analysis_results:
            mood_info = analysis_results.get('current_mood', {})
            if mood_info:
                prompt_parts.append(f"Current mood analysis: {mood_info.get('current_mood', 'Unknown')} (confidence: {mood_info.get('confidence', 0):.2f})")
        
        prompt_parts.append(f"User's current message: {user_message}")
        
        prompt_parts.append("\nPlease respond as a supportive therapist, acknowledging what the user has shared and helping them explore their feelings further. Keep your response concise but meaningful.")
        
        return "\n".join(prompt_parts)
    
    def generate_crisis_response(self, user_message: str) -> str:
        """Generate response for crisis situations"""
        try:
            self._rate_limit()
            
            system_prompt = """You are responding to someone who may be in a mental health crisis. Your response must:

1. Immediately express concern and care
2. Provide crisis resources and hotline numbers
3. Strongly encourage immediate professional help
4. Avoid making the situation worse
5. Be direct but compassionate

This is a crisis response - prioritize safety over everything else."""
            
            user_prompt = f"User message indicating potential crisis: {user_message}\n\nProvide an immediate, caring crisis response with resources."
            
            response = openai.ChatCompletion.create(
                model=self.default_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300,
                temperature=0.3  # Lower temperature for more consistent crisis responses
            )
            
            # Add crisis resources to any AI response
            crisis_resources = """

🆘 IMMEDIATE HELP:
• National Suicide Prevention Lifeline: 988
• Crisis Text Line: Text HOME to 741741
• Emergency Services: 911
• For immediate danger, go to your nearest emergency room"""
            
            return response.choices[0].message.content.strip() + crisis_resources
            
        except Exception as e:
            logger.error(f"Crisis response generation failed: {str(e)}")
            return """I'm very concerned about you. Please reach out for immediate help:

🆘 IMMEDIATE HELP:
• National Suicide Prevention Lifeline: 988
• Crisis Text Line: Text HOME to 741741
• Emergency Services: 911
• For immediate danger, go to your nearest emergency room

You don't have to go through this alone. There are people who want to help you."""
    
    def detect_crisis_keywords(self, text: str) -> bool:
        """Detect if text contains crisis keywords"""
        crisis_keywords = [
            'suicide', 'kill myself', 'end my life', 'want to die', 'better off dead',
            'no point living', 'end it all', 'hurt myself', 'self harm', 'overdose',
            'jump off', 'hanging', 'gun', 'pills', 'razor', 'cutting'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in crisis_keywords)
    
    def generate_initial_conversation_starter(self, analysis_results: Optional[Dict] = None) -> str:
        """Generate conversation starter based on analysis"""
        try:
            if analysis_results:
                mood = analysis_results.get('current_mood', {}).get('current_mood', 'neutral')
                risk_level = analysis_results.get('summary', {}).get('risk_level', 'minimal')
                primary_concern = analysis_results.get('summary', {}).get('primary_concern', 'None')
                
                prompt = f"""Generate a warm, welcoming conversation starter for someone who just completed a mental health analysis. 

Analysis results:
- Current mood: {mood}
- Risk level: {risk_level}
- Primary concern: {primary_concern}

Create a personalized, empathetic opening that:
1. Acknowledges their mood/state
2. Invites them to share more
3. Makes them feel heard and supported
4. Is conversational, not clinical

Keep it to 2-3 sentences maximum."""
                
                response = openai.ChatCompletion.create(
                    model=self.default_model,
                    messages=[
                        {"role": "system", "content": "You are a warm, empathetic therapist starting a conversation."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,
                    temperature=0.7
                )
                
                return response.choices[0].message.content.strip()
            
            else:
                # Default conversation starters
                default_starters = [
                    "Hello! I'm here to listen and support you. How are you feeling right now?",
                    "Welcome! I'm glad you're here. What's been on your mind lately?",
                    "Hi there! This is a safe space for you to share whatever you're experiencing. How has your day been?",
                    "Hello! I'm here to provide support and a listening ear. What would you like to talk about today?"
                ]
                
                import random
                return random.choice(default_starters)
                
        except Exception as e:
            logger.error(f"Failed to generate conversation starter: {str(e)}")
            return "Hello! I'm here to listen and support you. How are you feeling today?"

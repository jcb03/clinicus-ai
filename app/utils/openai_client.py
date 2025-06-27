from openai import OpenAI
import logging
from typing import Dict, List, Optional
import time
from config.settings import settings

logger = logging.getLogger(__name__)

class OpenAIClient:
    def __init__(self, api_key: str = None):
        """Initialize OpenAI client with robust error handling"""
        # FIXED: Use provided API key or get from settings
        self.api_key = api_key if api_key else settings.openai_api_key
        self.client = None
        self.is_connected = False
        
        # Debug API key loading
        logger.info(f"Attempting to initialize OpenAI with key length: {len(self.api_key) if self.api_key else 0}")
        
        # FIXED: Better API key validation
        if self.api_key and len(self.api_key) > 50 and self.api_key.startswith('sk-'):
            try:
                self.client = OpenAI(api_key=self.api_key)
                
                # Test the connection with a minimal request
                test_response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=3,
                    temperature=0.1
                )
                
                if test_response and test_response.choices:
                    self.is_connected = True
                    logger.info("✅ OpenAI client initialized and connection tested successfully")
                else:
                    raise Exception("Invalid response from OpenAI API")
                    
            except Exception as e:
                logger.error(f"❌ Failed to initialize OpenAI client: {e}")
                self.client = None
                self.is_connected = False
        else:
            logger.warning(f"❌ Invalid OpenAI API key. Length: {len(self.api_key) if self.api_key else 0}, Starts with sk-: {self.api_key.startswith('sk-') if self.api_key else False}")
            self.is_connected = False
        
        # Default parameters
        self.default_model = getattr(settings, 'openai_model', 'gpt-3.5-turbo')
        self.default_max_tokens = getattr(settings, 'max_tokens', 500)
        self.default_temperature = getattr(settings, 'temperature', 0.7)
        
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
                                 mood: str = "neutral",
                                 language: str = "en") -> str:
        """Generate therapist response with enhanced error handling and crisis detection"""
        
        # Check if client is available
        if not self.client or not self.is_connected:
            return self._fallback_response(user_message)
        
        try:
            self._rate_limit()
            
            # Enhanced crisis detection
            if self.detect_crisis_keywords(user_message):
                return self.generate_crisis_response(user_message)
            
            # Build enhanced system prompt
            system_prompt = self._build_therapist_system_prompt(analysis_results, mood)
            
            # Build user prompt with context
            user_prompt = self._build_therapist_user_prompt(user_message, conversation_context, analysis_results)
            
            # Make API call
            response = self.client.chat.completions.create(
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
            
            ai_response = response.choices[0].message.content.strip()
            
            # Validate response
            if not ai_response or len(ai_response) < 10:
                return self._fallback_response(user_message)
            
            logger.info("✅ OpenAI response generated successfully")
            return ai_response
            
        except Exception as e:
            logger.error(f"❌ OpenAI request failed: {str(e)}")
            return self._fallback_response(user_message)
    
    def _build_therapist_system_prompt(self, analysis_results: Optional[Dict], mood: str) -> str:
        """Build comprehensive system prompt for therapist responses"""
        
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
- If the user expresses thoughts of self-harm, immediately encourage them to seek emergency help
- Keep responses conversational and supportive, not clinical
- Ask follow-up questions to show engagement
- Provide practical, actionable advice when appropriate
- Limit responses to 100-150 words for better engagement"""
        
        # Add analysis-based context
        if analysis_results:
            risk_level = analysis_results.get('summary', {}).get('risk_level', 'minimal')
            primary_concern = analysis_results.get('summary', {}).get('primary_concern', 'None')
            
            if risk_level in ['moderate', 'high']:
                base_prompt += f"\n\nIMPORTANT: The user's recent analysis suggests {risk_level} risk level with primary concern: {primary_concern}. Be extra supportive and gently encourage professional consultation."
            
            if primary_concern and primary_concern not in ['None detected', 'none_detected']:
                base_prompt += f"\n\nThe user may be experiencing signs related to {primary_concern}. Tailor your response with appropriate sensitivity to this concern."
        
        # Add mood context
        if mood and mood.lower() not in ['neutral', 'unknown']:
            base_prompt += f"\n\nThe user's current mood appears to be: {mood}. Acknowledge this appropriately in your response."
        
        return base_prompt
    
    def _build_therapist_user_prompt(self, user_message: str, context: str, analysis_results: Optional[Dict]) -> str:
        """Build user prompt with enhanced context"""
        prompt_parts = []
        
        # Add conversation context
        if context and len(context.strip()) > 0:
            prompt_parts.append(f"Recent conversation context:\n{context}\n")
        
        # Add analysis context
        if analysis_results:
            # Mood information
            mood_info = analysis_results.get('current_mood', {})
            if mood_info:
                prompt_parts.append(f"Current mood analysis: {mood_info.get('current_mood', 'Unknown')} (confidence: {mood_info.get('confidence', 0):.2f})")
            
            # Primary concerns
            summary = analysis_results.get('summary', {})
            primary_concern = summary.get('primary_concern', 'None')
            if primary_concern not in ['None detected', 'none_detected']:
                prompt_parts.append(f"Primary concern detected: {primary_concern}")
        
        # Add user message
        prompt_parts.append(f"User's current message: {user_message}")
        prompt_parts.append("\nPlease respond as a supportive therapist, acknowledging what the user has shared and helping them explore their feelings further. Keep your response warm, conversational, and under 150 words. Ask a follow-up question to encourage continued dialogue.")
        
        return "\n".join(prompt_parts)
    
    def detect_crisis_keywords(self, text: str) -> bool:
        """Enhanced crisis keyword detection"""
        crisis_keywords = [
            # Suicide and self-harm keywords
            'suicide', 'kill myself', 'end my life', 'want to die', 'better off dead',
            'no point living', 'end it all', 'hurt myself', 'self harm', 'cut myself',
            'jump off', 'break my skull', 'fucking die', 'want to fucking die',
            'going to kill myself', 'plan to kill myself', 'thinking of killing myself',
            'overdose', 'hanging myself', 'gun to my head', 'razor blade',
            'slit my wrists', 'take pills', 'carbon monoxide', 'bridge jump',
            'train tracks', 'rope around neck', 'bullet to head'
        ]
        
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in crisis_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        if found_keywords:
            logger.warning(f"🚨 Crisis keywords detected: {found_keywords}")
            return True
        
        return False
    
    def generate_crisis_response(self, user_message: str) -> str:
        """Generate immediate crisis response"""
        try:
            if not self.client or not self.is_connected:
                return self._emergency_crisis_response()
            
            self._rate_limit()
            
            system_prompt = """You are responding to someone who is in a mental health crisis. Your response must:

1. Immediately express concern and care
2. Provide crisis resources and hotline numbers
3. Strongly encourage immediate professional help
4. Show hope and support
5. Be direct but compassionate

This is a crisis response - prioritize safety over everything else. Keep response under 200 words."""
            
            user_prompt = f"User message indicating crisis: {user_message}\n\nProvide an immediate, caring crisis response with resources."
            
            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300,
                temperature=0.2  # Lower temperature for more consistent crisis responses
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Add crisis resources to any AI response
            crisis_resources = self._get_crisis_resources()
            
            return ai_response + "\n\n" + crisis_resources
            
        except Exception as e:
            logger.error(f"❌ Crisis response generation failed: {str(e)}")
            return self._emergency_crisis_response()
    
    def _get_crisis_resources(self) -> str:
        """Get crisis resources"""
        return """🆘 IMMEDIATE HELP:
• National Suicide Prevention Lifeline: 988
• Crisis Text Line: Text HOME to 741741
• Emergency Services: 911
• International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/
• For immediate danger, go to your nearest emergency room

You are not alone. Help is available 24/7."""
    
    def _emergency_crisis_response(self) -> str:
        """Emergency crisis response when OpenAI is unavailable"""
        response = """I'm very concerned about you. Please reach out for immediate help:"""
        return response + "\n\n" + self._get_crisis_resources()
    
    def _fallback_response(self, user_message: str) -> str:
        """Fallback response when OpenAI is unavailable"""
        
        # Check if crisis - provide emergency response even without OpenAI
        if self.detect_crisis_keywords(user_message):
            return self._emergency_crisis_response()
        
        # Regular fallback responses
        fallback_responses = [
            "I'm here to listen and support you. How are you feeling right now?",
            "I want to understand what you're going through. Can you tell me more about how you're feeling?",
            "Your feelings matter, and I'm here to listen. What's been on your mind lately?",
            "I'm here to support you through this. What would be most helpful for you to talk about right now?",
            "Thank you for sharing with me. How are you coping with everything you're going through?",
            "I hear you, and I want to help. Can you tell me more about what's been challenging for you?",
            "I'm here to listen without judgment. What's been weighing on your mind?",
            "Your experiences are important. How can I best support you right now?"
        ]
        
        # Simple selection based on message content
        if any(word in user_message.lower() for word in ['sad', 'depressed', 'down']):
            return "I can hear that you're going through a difficult time. Your feelings are valid, and I'm here to listen. What's been making you feel this way?"
        elif any(word in user_message.lower() for word in ['anxious', 'worried', 'nervous']):
            return "It sounds like you're feeling anxious or worried about something. That can be really overwhelming. What's been on your mind that's causing these feelings?"
        elif any(word in user_message.lower() for word in ['angry', 'frustrated', 'mad']):
            return "I can sense there's some frustration or anger you're dealing with. Those are completely valid feelings. What's been triggering these emotions for you?"
        else:
            import random
            return random.choice(fallback_responses)
    
    def generate_initial_conversation_starter(self, analysis_results: Optional[Dict] = None) -> str:
        """Generate conversation starter based on analysis"""
        try:
            if not self.client or not self.is_connected:
                return self._default_conversation_starter(analysis_results)
            
            if analysis_results and analysis_results.get('summary', {}).get('primary_concern') not in ['None detected', 'none_detected']:
                mood = analysis_results.get('current_mood', {}).get('current_mood', 'neutral')
                risk_level = analysis_results.get('summary', {}).get('risk_level', 'minimal')
                primary_concern = analysis_results.get('summary', {}).get('primary_concern', 'None')
                
                prompt = f"""Generate a warm, welcoming conversation starter for someone who just completed a mental health analysis. 

Analysis results:
- Current mood: {mood}
- Risk level: {risk_level}
- Primary concern: {primary_concern}

Create a personalized, empathetic opening that:
1. Acknowledges their current state without being clinical
2. Shows genuine care and concern
3. Invites them to share more
4. Makes them feel heard and supported
5. Is conversational and warm

Keep it to 2-3 sentences maximum. Be supportive but not overwhelming."""
                
                response = self.client.chat.completions.create(
                    model=self.default_model,
                    messages=[
                        {"role": "system", "content": "You are a warm, empathetic therapist starting a conversation with someone who needs support."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,
                    temperature=0.7
                )
                
                return response.choices[0].message.content.strip()
            
            else:
                return self._default_conversation_starter(analysis_results)
                
        except Exception as e:
            logger.error(f"❌ Failed to generate conversation starter: {str(e)}")
            return self._default_conversation_starter(analysis_results)
    
    def _default_conversation_starter(self, analysis_results: Optional[Dict] = None) -> str:
        """Default conversation starters when OpenAI is unavailable"""
        
        if analysis_results:
            primary_concern = analysis_results.get('summary', {}).get('primary_concern', 'None')
            if primary_concern not in ['None detected', 'none_detected']:
                return f"Hello! I'm here to listen and support you. I noticed you might be dealing with some challenges related to {primary_concern.lower()}. How are you feeling right now?"
        
        # Default starters
        default_starters = [
            "Hello! I'm here to listen and support you. How are you feeling right now?",
            "Welcome! I'm glad you're here. What's been on your mind lately?",
            "Hi there! This is a safe space for you to share whatever you're experiencing. How has your day been?",
            "Hello! I'm here to provide support and a listening ear. What would you like to talk about today?",
            "I'm here to listen without judgment. How are you doing today?"
        ]
        
        import random
        return random.choice(default_starters)

    def test_connection(self) -> bool:
        """Test OpenAI connection"""
        try:
            if not self.client:
                return False
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
                temperature=0
            )
            
            return bool(response and response.choices)
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False

# Test function
if __name__ == "__main__":
    print("Testing OpenAI Client...")
    client = OpenAIClient()
    print(f"OpenAI client connected: {client.is_connected}")
    
    if client.is_connected:
        test_response = client.generate_therapist_response("I'm feeling sad today")
        print(f"✅ Test response: {test_response}")
    else:
        fallback_response = client._fallback_response("I'm feeling sad today")
        print(f"⚠️ Fallback response: {fallback_response}")

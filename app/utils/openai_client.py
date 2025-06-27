from openai import OpenAI
import logging
from typing import Dict, List, Optional
import time
from config.settings import settings

logger = logging.getLogger(__name__)

class OpenAIClient:
    def __init__(self, api_key: str = None):
        """Initialize OpenAI client with robust error handling"""
        self.api_key = api_key or settings.openai_api_key
        self.client = None
        self.is_connected = False
        
        # Better API key validation
        if self.api_key and len(self.api_key) > 20 and self.api_key.startswith('sk-'):
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
                    logger.info("OpenAI client initialized and connection tested successfully")
                else:
                    raise Exception("Invalid response from OpenAI API")
                    
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
                self.is_connected = False
        else:
            logger.warning(f"Invalid OpenAI API key format. Key length: {len(self.api_key) if self.api_key else 0}")
            self.is_connected = False
        
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
                                 mood: str = "neutral",
                                 language: str = "en") -> str:
        """Generate therapist response with enhanced error handling and crisis detection"""
        
        # Check if client is available
        if not self.client or not self.is_connected:
            return self._fallback_response(user_message, language)
        
        try:
            self._rate_limit()
            
            # Enhanced crisis detection
            if self.detect_crisis_keywords(user_message):
                return self.generate_crisis_response(user_message, language)
            
            # Build enhanced system prompt
            system_prompt = self._build_enhanced_therapist_system_prompt(analysis_results, mood, language)
            
            # Build user prompt with context
            user_prompt = self._build_therapist_user_prompt(user_message, conversation_context, analysis_results, language)
            
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
                return self._fallback_response(user_message, language)
            
            logger.info("OpenAI response generated successfully")
            return ai_response
            
        except Exception as e:
            logger.error(f"OpenAI request failed: {str(e)}")
            return self._fallback_response(user_message, language)
    
    def _build_enhanced_therapist_system_prompt(self, analysis_results: Optional[Dict], mood: str, language: str) -> str:
        """Build comprehensive system prompt for therapist responses"""
        
        if language == 'hi':
            base_prompt = """आप एक करुणामय, पेशेवर AI थेरेपिस्ट सहायक हैं। आपकी भूमिका है:

1. सहानुभूतिपूर्ण, सहायक प्रतिक्रियाएं प्रदान करना
2. उपयोगकर्ताओं को अपनी भावनाओं को समझने में मदद करने के लिए विचारशील प्रश्न पूछना
3. कोमल मार्गदर्शन और मुकाबला रणनीतियां प्रदान करना
4. गंभीर चिंताओं के लिए हमेशा पेशेवर मदद को प्रोत्साहित करना
5. उचित सीमाएं बनाए रखना
6. कभी भी निदान न करना या चिकित्सा सलाह न देना

दिशानिर्देश:
- गर्मजोशी, समझदार और गैर-न्यायिक रहें
- सक्रिय सुनने की तकनीकों का उपयोग करें
- उपयोगकर्ता की भावनाओं को मान्य करें
- उपयुक्त होने पर स्वस्थ मुकाबला तंत्र सुझाएं
- यदि उपयोगकर्ता आत्म-हानि के विचार व्यक्त करता है, तो तुरंत आपातकालीन सहायता लेने के लिए प्रोत्साहित करें"""
        else:
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
- Provide practical, actionable advice when appropriate"""
        
        # Add analysis-based context
        if analysis_results:
            risk_level = analysis_results.get('summary', {}).get('risk_level', 'minimal')
            primary_concern = analysis_results.get('summary', {}).get('primary_concern', 'None')
            
            if risk_level in ['moderate', 'high']:
                if language == 'hi':
                    base_prompt += f"\n\nमहत्वपूर्ण: उपयोगकर्ता के हाल के विश्लेषण से {risk_level} जोखिम स्तर का संकेत मिलता है जिसकी मुख्य चिंता है: {primary_concern}। अतिरिक्त सहायक बनें और कोमलता से पेशेवर परामर्श को प्रोत्साहित करें।"
                else:
                    base_prompt += f"\n\nIMPORTANT: The user's recent analysis suggests {risk_level} risk level with primary concern: {primary_concern}. Be extra supportive and gently encourage professional consultation."
            
            if primary_concern and primary_concern not in ['None detected', 'none_detected']:
                if language == 'hi':
                    base_prompt += f"\n\nउपयोगकर्ता {primary_concern} से संबंधित संकेतों का अनुभव कर रहा हो सकता है। इस चिंता के लिए उचित संवेदनशीलता के साथ अपनी प्रतिक्रिया तैयार करें।"
                else:
                    base_prompt += f"\n\nThe user may be experiencing signs related to {primary_concern}. Tailor your response with appropriate sensitivity to this concern."
        
        # Add mood context
        if mood and mood.lower() not in ['neutral', 'unknown']:
            if language == 'hi':
                base_prompt += f"\n\nउपयोगकर्ता का वर्तमान मूड प्रतीत होता है: {mood}। अपनी प्रतिक्रिया में इसे उचित रूप से स्वीकार करें।"
            else:
                base_prompt += f"\n\nThe user's current mood appears to be: {mood}. Acknowledge this appropriately in your response."
        
        return base_prompt
    
    def _build_therapist_user_prompt(self, user_message: str, context: str, analysis_results: Optional[Dict], language: str) -> str:
        """Build user prompt with enhanced context"""
        prompt_parts = []
        
        # Add conversation context
        if context and len(context.strip()) > 0:
            if language == 'hi':
                prompt_parts.append(f"हाल की बातचीत का संदर्भ:\n{context}\n")
            else:
                prompt_parts.append(f"Recent conversation context:\n{context}\n")
        
        # Add analysis context
        if analysis_results:
            # Mood information
            mood_info = analysis_results.get('current_mood', {})
            if mood_info:
                if language == 'hi':
                    prompt_parts.append(f"वर्तमान मूड विश्लेषण: {mood_info.get('current_mood', 'अज्ञात')} (आत्मविश्वास: {mood_info.get('confidence', 0):.2f})")
                else:
                    prompt_parts.append(f"Current mood analysis: {mood_info.get('current_mood', 'Unknown')} (confidence: {mood_info.get('confidence', 0):.2f})")
            
            # Primary concerns
            summary = analysis_results.get('summary', {})
            primary_concern = summary.get('primary_concern', 'None')
            if primary_concern not in ['None detected', 'none_detected']:
                if language == 'hi':
                    prompt_parts.append(f"मुख्य चिंता: {primary_concern}")
                else:
                    prompt_parts.append(f"Primary concern detected: {primary_concern}")
        
        # Add user message
        if language == 'hi':
            prompt_parts.append(f"उपयोगकर्ता का वर्तमान संदेश: {user_message}")
            prompt_parts.append("\nकृपया एक सहायक थेरेपिस्ट के रूप में जवाब दें, जो उपयोगकर्ता ने साझा किया है उसे स्वीकार करते हुए और उन्हें अपनी भावनाओं को और समझने में मदद करते हुए। अपनी प्रतिक्रिया संक्षिप्त लेकिन अर्थपूर्ण रखें। हिंदी में जवाब दें।")
        else:
            prompt_parts.append(f"User's current message: {user_message}")
            prompt_parts.append("\nPlease respond as a supportive therapist, acknowledging what the user has shared and helping them explore their feelings further. Keep your response warm, conversational, and under 150 words. Ask a follow-up question to encourage continued dialogue.")
        
        return "\n".join(prompt_parts)
    
    def detect_crisis_keywords(self, text: str) -> bool:
        """Enhanced crisis keyword detection"""
        crisis_keywords = [
            # English crisis keywords
            'suicide', 'kill myself', 'end my life', 'want to die', 'better off dead',
            'no point living', 'end it all', 'hurt myself', 'self harm', 'cut myself',
            'jump off', 'break my skull', 'fucking die', 'want to fucking die',
            'going to kill myself', 'plan to kill myself', 'thinking of killing myself',
            'overdose', 'hanging myself', 'gun to my head', 'razor blade',
            
            # Hindi crisis keywords
            'आत्महत्या', 'खुद को मार', 'जीना नहीं चाहता', 'मर जाना चाहता', 'खुदकुशी',
            'अपने आप को मार', 'जीवन समाप्त कर', 'मरना चाहता हूं'
        ]
        
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in crisis_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        if found_keywords:
            logger.warning(f"Crisis keywords detected: {found_keywords}")
            return True
        
        return False
    
    def generate_crisis_response(self, user_message: str, language: str = "en") -> str:
        """Generate immediate crisis response"""
        try:
            if not self.client or not self.is_connected:
                return self._emergency_crisis_response(language)
            
            self._rate_limit()
            
            if language == 'hi':
                system_prompt = """आप किसी ऐसे व्यक्ति को जवाब दे रहे हैं जो मानसिक स्वास्थ्य संकट में है। आपकी प्रतिक्रिया में होना चाहिए:

1. तुरंत चिंता और देखभाल व्यक्त करना
2. संकट संसाधन और हेल्पलाइन नंबर प्रदान करना (भारतीय संख्याएं)
3. तत्काल पेशेवर मदद के लिए दृढ़ता से प्रोत्साहित करना
4. उम्मीद और सहारा दिखाना
5. प्रत्यक्ष लेकिन दयालु होना

यह एक संकट प्रतिक्रिया है - सुरक्षा को सर्वोच्च प्राथमिकता दें।"""
                
                user_prompt = f"उपयोगकर्ता संदेश जो संकट का संकेत देता है: {user_message}\n\nभारतीय संसाधनों के साथ तत्काल, देखभाल करने वाली संकट प्रतिक्रिया प्रदान करें।"
            else:
                system_prompt = """You are responding to someone who is in a mental health crisis. Your response must:

1. Immediately express concern and care
2. Provide crisis resources and hotline numbers
3. Strongly encourage immediate professional help
4. Show hope and support
5. Be direct but compassionate

This is a crisis response - prioritize safety over everything else."""
                
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
            crisis_resources = self._get_crisis_resources(language)
            
            return ai_response + "\n\n" + crisis_resources
            
        except Exception as e:
            logger.error(f"Crisis response generation failed: {str(e)}")
            return self._emergency_crisis_response(language)
    
    def _get_crisis_resources(self, language: str) -> str:
        """Get crisis resources based on language"""
        if language == 'hi':
            return """🆘 तत्काल सहायता:
• आपातकालीन सेवाएं: 112
• राष्ट्रीय हेल्पलाइन: 9152987821
• AASRA (मुंबई): 9820466726
• Sumaitri (दिल्ली): 011-23389090
• तत्काल खतरे के लिए, अपने निकटतम आपातकालीन कक्ष में जाएं

आप अकेले नहीं हैं। मदद उपलब्ध है।"""
        else:
            return """🆘 IMMEDIATE HELP:
• National Suicide Prevention Lifeline: 988
• Crisis Text Line: Text HOME to 741741
• Emergency Services: 911
• International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/
• For immediate danger, go to your nearest emergency room

You are not alone. Help is available."""
    
    def _emergency_crisis_response(self, language: str) -> str:
        """Emergency crisis response when OpenAI is unavailable"""
        if language == 'hi':
            response = """मुझे आपकी बहुत चिंता है। कृपया तत्काल सहायता के लिए संपर्क करें:"""
        else:
            response = """I'm very concerned about you. Please reach out for immediate help:"""
        
        return response + "\n\n" + self._get_crisis_resources(language)
    
    def _fallback_response(self, user_message: str, language: str) -> str:
        """Fallback response when OpenAI is unavailable"""
        
        # Check if crisis - provide emergency response even without OpenAI
        if self.detect_crisis_keywords(user_message):
            return self._emergency_crisis_response(language)
        
        # Regular fallback responses
        if language == 'hi':
            fallback_responses = [
                "मैं यहाँ आपकी बात सुनने के लिए हूँ। आप कैसा महसूस कर रहे हैं?",
                "आप जो भी साझा करना चाहते हैं, मैं यहाँ हूँ। आपकी भावनाओं के बारे में और बताएं।",
                "मैं समझना चाहता हूँ कि आप क्या अनुभव कर रहे हैं। कृपया और बताएं।",
                "आपकी भावनाएं मायने रखती हैं। आप इस समय कैसा महसूस कर रहे हैं?"
            ]
        else:
            fallback_responses = [
                "I'm here to listen and support you. How are you feeling right now?",
                "I want to understand what you're going through. Can you tell me more about how you're feeling?",
                "Your feelings matter, and I'm here to listen. What's been on your mind lately?",
                "I'm here to support you through this. What would be most helpful for you to talk about right now?",
                "Thank you for sharing with me. How are you coping with everything you're going through?",
                "I hear you, and I want to help. Can you tell me more about what's been challenging for you?"
            ]
        
        # Simple selection based on message length
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
            logger.error(f"Failed to generate conversation starter: {str(e)}")
            return self._default_conversation_starter(analysis_results)
    
    def _default_conversation_starter(self, analysis_results: Optional[Dict] = None) -> str:
        """Default conversation starters when OpenAI is unavailable"""
        
        if analysis_results:
            primary_concern = analysis_results.get('summary', {}).get('primary_concern', 'None')
            if primary_concern not in ['None detected', 'none_detected']:
                return f"Hello! I'm here to listen and support you. I noticed you might be dealing with some challenges related to {primary_concern}. How are you feeling right now?"
        
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
            logger.error(f"Connection test failed: {e}")
            return False

# Test function
if __name__ == "__main__":
    client = OpenAIClient()
    print(f"OpenAI client connected: {client.is_connected}")
    
    if client.is_connected:
        test_response = client.generate_therapist_response("I'm feeling sad today")
        print(f"Test response: {test_response}")
    else:
        fallback_response = client._fallback_response("I'm feeling sad today", "en")
        print(f"Fallback response: {fallback_response}")

import openai
from openai import OpenAI
import logging
from typing import Dict, List, Optional
import time
import json
from config.settings import settings

logger = logging.getLogger(__name__)

class OpenAIClient:
    def __init__(self, api_key: str = None):
        """Initialize OpenAI client with new API"""
        self.api_key = api_key or settings.openai_api_key
        self.client = OpenAI(api_key=self.api_key)
        
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
                                 language: str = "en") -> str:  # Added language parameter
        """Generate therapist response with language support"""
        try:
            self._rate_limit()
            
            # Build context-aware prompt
            system_prompt = self._build_therapist_system_prompt(analysis_results, mood, language)
            user_prompt = self._build_therapist_user_prompt(user_message, conversation_context, analysis_results, language)
            
            # Use new OpenAI API syntax
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
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI request failed: {str(e)}")
            if language == 'hi':
                return "मुझे खेद है, मुझे अभी जवाब देने में परेशानी हो रही है। आप कैसा महसूस कर रहे हैं, और क्या कोई विशेष बात है जिसके बारे में आप बात करना चाहते हैं?"
            else:
                return "I'm sorry, I'm having difficulty responding right now. How are you feeling, and is there anything specific you'd like to talk about?"
    
    def _build_therapist_system_prompt(self, analysis_results: Optional[Dict], mood: str, language: str) -> str:
        """Build system prompt for therapist responses with language support"""
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
- अपनी प्रतिक्रियाओं में सक्रिय सुनने की तकनीकों का उपयोग करें
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
- If the user expresses thoughts of self-harm, immediately encourage them to seek emergency help"""

        # Add context-specific guidance
        if analysis_results:
            risk_level = analysis_results.get('summary', {}).get('risk_level', 'minimal')
            primary_concern = analysis_results.get('summary', {}).get('primary_concern', 'None')
            
            if risk_level in ['moderate', 'high']:
                if language == 'hi':
                    base_prompt += f"\n\nमहत्वपूर्ण: उपयोगकर्ता के हाल के विश्लेषण से {risk_level} जोखिम स्तर का संकेत मिलता है जिसकी मुख्य चिंता है: {primary_concern}। अतिरिक्त सहायक बनें और कोमलता से पेशेवर परामर्श को प्रोत्साहित करें।"
                else:
                    base_prompt += f"\n\nIMPORTANT: The user's recent analysis suggests {risk_level} risk level with primary concern: {primary_concern}. Be extra supportive and gently encourage professional consultation."
            
            if primary_concern and primary_concern != 'None detected':
                if language == 'hi':
                    base_prompt += f"\n\nउपयोगकर्ता {primary_concern} से संबंधित संकेतों का अनुभव कर रहा हो सकता है। इस चिंता के लिए उचित संवेदनशीलता के साथ अपनी प्रतिक्रिया तैयार करें।"
                else:
                    base_prompt += f"\n\nThe user may be experiencing signs related to {primary_concern}. Tailor your response with appropriate sensitivity to this concern."
        
        if mood and mood.lower() != 'neutral':
            if language == 'hi':
                base_prompt += f"\n\nउपयोगकर्ता का वर्तमान मूड प्रतीत होता है: {mood}। अपनी प्रतिक्रिया में इसे उचित रूप से स्वीकार करें।"
            else:
                base_prompt += f"\n\nThe user's current mood appears to be: {mood}. Acknowledge this appropriately in your response."
        
        return base_prompt
    
    def _build_therapist_user_prompt(self, user_message: str, context: str, analysis_results: Optional[Dict], language: str) -> str:
        """Build user prompt with context and language support"""
        prompt_parts = []
        
        if context:
            if language == 'hi':
                prompt_parts.append(f"हाल की बातचीत का संदर्भ:\n{context}\n")
            else:
                prompt_parts.append(f"Recent conversation context:\n{context}\n")
        
        if analysis_results:
            mood_info = analysis_results.get('current_mood', {})
            if mood_info:
                if language == 'hi':
                    prompt_parts.append(f"वर्तमान मूड विश्लेषण: {mood_info.get('current_mood', 'अज्ञात')} (आत्मविश्वास: {mood_info.get('confidence', 0):.2f})")
                else:
                    prompt_parts.append(f"Current mood analysis: {mood_info.get('current_mood', 'Unknown')} (confidence: {mood_info.get('confidence', 0):.2f})")
        
        if language == 'hi':
            prompt_parts.append(f"उपयोगकर्ता का वर्तमान संदेश: {user_message}")
            prompt_parts.append("\nकृपया एक सहायक थेरेपिस्ट के रूप में जवाब दें, जो उपयोगकर्ता ने साझा किया है उसे स्वीकार करते हुए और उन्हें अपनी भावनाओं को और समझने में मदद करते हुए। अपनी प्रतिक्रिया संक्षिप्त लेकिन अर्थपूर्ण रखें। हिंदी में जवाब दें।")
        else:
            prompt_parts.append(f"User's current message: {user_message}")
            prompt_parts.append("\nPlease respond as a supportive therapist, acknowledging what the user has shared and helping them explore their feelings further. Keep your response concise but meaningful.")
        
        return "\n".join(prompt_parts)
    
    def detect_crisis_keywords(self, text: str) -> bool:
        """Detect if text contains crisis keywords"""
        crisis_keywords = [
            'suicide', 'kill myself', 'end my life', 'want to die', 'better off dead',
            'no point living', 'end it all', 'hurt myself', 'self harm', 'overdose',
            'jump off', 'hanging', 'gun', 'pills', 'razor', 'cutting',
            # Hindi crisis keywords
            'आत्महत्या', 'खुद को मार', 'जीना नहीं चाहता', 'मर जाना चाहता', 'खुदकुशी'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in crisis_keywords)
    
    def generate_crisis_response(self, user_message: str, language: str = "en") -> str:
        """Generate response for crisis situations"""
        try:
            self._rate_limit()
            
            if language == 'hi':
                system_prompt = """आप किसी ऐसे व्यक्ति को जवाब दे रहे हैं जो मानसिक स्वास्थ्य संकट में हो सकता है। आपकी प्रतिक्रिया में होना चाहिए:

1. तुरंत चिंता और देखभाल व्यक्त करना
2. संकट संसाधन और हेल्पलाइन नंबर प्रदान करना
3. तत्काल पेशेवर मदद के लिए दृढ़ता से प्रोत्साहित करना
4. स्थिति को बदतर न बनाना
5. प्रत्यक्ष लेकिन दयालु होना

यह एक संकट प्रतिक्रिया है - सब कुछ के ऊपर सुरक्षा को प्राथमिकता दें।"""
                
                user_prompt = f"उपयोगकर्ता संदेश जो संभावित संकट का संकेत देता है: {user_message}\n\nसंसाधनों के साथ तत्काल, देखभाल करने वाली संकट प्रतिक्रिया प्रदान करें।"
            else:
                system_prompt = """You are responding to someone who may be in a mental health crisis. Your response must:

1. Immediately express concern and care
2. Provide crisis resources and hotline numbers
3. Strongly encourage immediate professional help
4. Avoid making the situation worse
5. Be direct but compassionate

This is a crisis response - prioritize safety over everything else."""
                
                user_prompt = f"User message indicating potential crisis: {user_message}\n\nProvide an immediate, caring crisis response with resources."
            
            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300,
                temperature=0.3  # Lower temperature for more consistent crisis responses
            )
            
            # Add crisis resources to any AI response
            if language == 'hi':
                crisis_resources = """

🆘 तत्काल सहायता:
• आपातकालीन सेवाएं: 112
• राष्ट्रीय हेल्पलाइन: 9152987821
• तत्काल खतरे के लिए, अपने निकटतम आपातकालीन कक्ष में जाएं"""
            else:
                crisis_resources = """

🆘 IMMEDIATE HELP:
• National Suicide Prevention Lifeline: 988
• Crisis Text Line: Text HOME to 741741
• Emergency Services: 911
• For immediate danger, go to your nearest emergency room"""
            
            return response.choices[0].message.content.strip() + crisis_resources
            
        except Exception as e:
            logger.error(f"Crisis response generation failed: {str(e)}")
            if language == 'hi':
                return """मुझे आपकी बहुत चिंता है। कृपया तत्काल सहायता के लिए संपर्क करें:

🆘 तत्काल सहायता:
• आपातकालीन सेवाएं: 112
• राष्ट्रीय हेल्पलाइन: 9152987821
• तत्काल खतरे के लिए, अपने निकटतम आपातकालीन कक्ष में जाएं

आपको अकेले इससे गुजरना नहीं है। ऐसे लोग हैं जो आपकी मदद करना चाहते हैं।"""
            else:
                return """I'm very concerned about you. Please reach out for immediate help:

🆘 IMMEDIATE HELP:
• National Suicide Prevention Lifeline: 988
• Crisis Text Line: Text HOME to 741741
• Emergency Services: 911
• For immediate danger, go to your nearest emergency room

You don't have to go through this alone. There are people who want to help you."""
    
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
                
                response = self.client.chat.completions.create(
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

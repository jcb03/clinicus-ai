from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import logging
import streamlit as st

logger = logging.getLogger(__name__)

class ConversationManager:
    def __init__(self):
        """Initialize conversation manager"""
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state for conversations"""
        if 'conversations' not in st.session_state:
            st.session_state.conversations = []
        
        if 'current_conversation' not in st.session_state:
            st.session_state.current_conversation = self._create_new_conversation()
        
        if 'conversation_counter' not in st.session_state:
            st.session_state.conversation_counter = 1
    
    def _create_new_conversation(self) -> Dict:
        """Create a new conversation structure"""
        return {
            'id': f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'messages': [],
            'analysis_results': None,
            'mood_timeline': [],
            'start_time': datetime.now(),
            'last_activity': datetime.now(),
            'session_summary': {
                'total_messages': 0,
                'user_messages': 0,
                'ai_messages': 0,
                'primary_emotions': [],
                'risk_assessments': []
            }
        }
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Add message to current conversation"""
        try:
            message = {
                'role': role,  # 'user' or 'assistant'
                'content': content,
                'timestamp': datetime.now(),
                'metadata': metadata or {}
            }
            
            st.session_state.current_conversation['messages'].append(message)
            st.session_state.current_conversation['last_activity'] = datetime.now()
            
            # Update session summary
            st.session_state.current_conversation['session_summary']['total_messages'] += 1
            if role == 'user':
                st.session_state.current_conversation['session_summary']['user_messages'] += 1
            elif role == 'assistant':
                st.session_state.current_conversation['session_summary']['ai_messages'] += 1
            
            logger.info(f"Message added: {role}")
            
        except Exception as e:
            logger.error(f"Failed to add message: {str(e)}")
    
    def add_analysis_result(self, analysis: Dict) -> None:
        """Add analysis result to current conversation"""
        try:
            st.session_state.current_conversation['analysis_results'] = {
                'analysis': analysis,
                'timestamp': datetime.now()
            }
            
            # Add to mood timeline
            if 'current_mood' in analysis:
                mood_entry = {
                    'mood': analysis['current_mood']['current_mood'],
                    'confidence': analysis['current_mood']['confidence'],
                    'timestamp': datetime.now()
                }
                st.session_state.current_conversation['mood_timeline'].append(mood_entry)
            
            # Add to risk assessments
            if 'summary' in analysis and 'risk_level' in analysis['summary']:
                risk_entry = {
                    'risk_level': analysis['summary']['risk_level'],
                    'primary_concern': analysis['summary'].get('primary_concern', 'None'),
                    'timestamp': datetime.now()
                }
                st.session_state.current_conversation['session_summary']['risk_assessments'].append(risk_entry)
            
            logger.info("Analysis result added to conversation")
            
        except Exception as e:
            logger.error(f"Failed to add analysis result: {str(e)}")
    
    def get_current_messages(self) -> List[Dict]:
        """Get messages from current conversation"""
        return st.session_state.current_conversation.get('messages', [])
    
    def get_last_n_messages(self, n: int = 5) -> List[Dict]:
        """Get last n messages from current conversation"""
        messages = self.get_current_messages()
        return messages[-n:] if len(messages) > n else messages
    
    def get_conversation_context(self, max_messages: int = 10) -> str:
        """Get conversation context for AI responses"""
        try:
            messages = self.get_last_n_messages(max_messages)
            context_parts = []
            
            for msg in messages:
                role = "User" if msg['role'] == 'user' else "Assistant"
                content = msg['content'][:200]  # Limit content length
                context_parts.append(f"{role}: {content}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Failed to get conversation context: {str(e)}")
            return ""
    
    def save_current_conversation(self) -> None:
        """Save current conversation to history"""
        try:
            if st.session_state.current_conversation['messages']:
                # Update final summary
                self._update_conversation_summary()
                
                # Add to conversations history
                st.session_state.conversations.append(
                    st.session_state.current_conversation.copy()
                )
                
                # Start new conversation
                st.session_state.conversation_counter += 1
                st.session_state.current_conversation = self._create_new_conversation()
                
                logger.info("Conversation saved and new one started")
            
        except Exception as e:
            logger.error(f"Failed to save conversation: {str(e)}")
    
    def _update_conversation_summary(self) -> None:
        """Update conversation summary with final statistics"""
        try:
            conversation = st.session_state.current_conversation
            
            # Calculate duration
            start_time = conversation['start_time']
            end_time = conversation['last_activity']
            duration = (end_time - start_time).total_seconds() / 60  # minutes
            
            # Extract primary emotions from mood timeline
            moods = [entry['mood'] for entry in conversation['mood_timeline']]
            if moods:
                # Get most common mood
                mood_counts = {}
                for mood in moods:
                    mood_counts[mood] = mood_counts.get(mood, 0) + 1
                primary_mood = max(mood_counts, key=mood_counts.get)
            else:
                primary_mood = 'Unknown'
            
            # Update summary
            conversation['session_summary'].update({
                'duration_minutes': round(duration, 1),
                'primary_mood': primary_mood,
                'mood_changes': len(conversation['mood_timeline']),
                'end_time': end_time
            })
            
        except Exception as e:
            logger.error(f"Failed to update conversation summary: {str(e)}")
    
    def get_all_conversations(self) -> List[Dict]:
        """Get all conversation history"""
        return st.session_state.conversations
    
    def get_conversation_by_id(self, conversation_id: str) -> Optional[Dict]:
        """Get specific conversation by ID"""
        for conv in st.session_state.conversations:
            if conv.get('id') == conversation_id:
                return conv
        return None
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete conversation by ID"""
        try:
            st.session_state.conversations = [
                conv for conv in st.session_state.conversations 
                if conv.get('id') != conversation_id
            ]
            return True
        except Exception as e:
            logger.error(f"Failed to delete conversation: {str(e)}")
            return False
    
    def clear_all_conversations(self) -> None:
        """Clear all conversation history"""
        try:
            st.session_state.conversations = []
            st.session_state.current_conversation = self._create_new_conversation()
            st.session_state.conversation_counter = 1
            logger.info("All conversations cleared")
        except Exception as e:
            logger.error(f"Failed to clear conversations: {str(e)}")
    
    def get_conversation_statistics(self) -> Dict:
        """Get overall conversation statistics"""
        try:
            all_convs = st.session_state.conversations
            current_conv = st.session_state.current_conversation
            
            total_conversations = len(all_convs) + (1 if current_conv['messages'] else 0)
            total_messages = sum(conv['session_summary']['total_messages'] for conv in all_convs)
            total_messages += current_conv['session_summary']['total_messages']
            
            # Mood statistics
            all_moods = []
            for conv in all_convs:
                all_moods.extend([entry['mood'] for entry in conv.get('mood_timeline', [])])
            
            all_moods.extend([entry['mood'] for entry in current_conv.get('mood_timeline', [])])
            
            mood_distribution = {}
            for mood in all_moods:
                mood_distribution[mood] = mood_distribution.get(mood, 0) + 1
            
            return {
                'total_conversations': total_conversations,
                'total_messages': total_messages,
                'mood_distribution': mood_distribution,
                'most_common_mood': max(mood_distribution, key=mood_distribution.get) if mood_distribution else 'None',
                'active_since': min([conv['start_time'] for conv in all_convs] + [current_conv['start_time']]).isoformat() if all_convs or current_conv['messages'] else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get conversation statistics: {str(e)}")
            return {
                'total_conversations': 0,
                'total_messages': 0,
                'mood_distribution': {},
                'most_common_mood': 'None',
                'active_since': None
            }
    
    def format_conversation_for_display(self, conversation: Dict) -> Dict:
        """Format conversation for display in UI"""
        try:
            start_time = conversation['start_time']
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time)
            
            summary = conversation.get('session_summary', {})
            
            return {
                'id': conversation.get('id', 'Unknown'),
                'title': f"Session {start_time.strftime('%m/%d %H:%M')}",
                'start_time': start_time.strftime('%B %d, %Y at %I:%M %p'),
                'duration': f"{summary.get('duration_minutes', 0):.1f} min",
                'messages': summary.get('total_messages', 0),
                'primary_mood': summary.get('primary_mood', 'Unknown'),
                'risk_level': self._get_latest_risk_level(conversation),
                'preview': self._get_conversation_preview(conversation)
            }
            
        except Exception as e:
            logger.error(f"Failed to format conversation: {str(e)}")
            return {
                'id': 'error',
                'title': 'Error loading conversation',
                'start_time': 'Unknown',
                'duration': '0 min',
                'messages': 0,
                'primary_mood': 'Unknown',
                'risk_level': 'Unknown',
                'preview': 'Error loading preview'
            }
    
    def _get_latest_risk_level(self, conversation: Dict) -> str:
        """Get latest risk level from conversation"""
        try:
            risk_assessments = conversation.get('session_summary', {}).get('risk_assessments', [])
            if risk_assessments:
                return risk_assessments[-1]['risk_level']
            return 'Unknown'
        except:
            return 'Unknown'
    
    def _get_conversation_preview(self, conversation: Dict) -> str:
        """Get preview text from conversation"""
        try:
            messages = conversation.get('messages', [])
            user_messages = [msg for msg in messages if msg['role'] == 'user']
            
            if user_messages:
                last_user_message = user_messages[-1]['content']
                return last_user_message[:100] + "..." if len(last_user_message) > 100 else last_user_message
            
            return "No user messages"
            
        except:
            return "Preview unavailable"

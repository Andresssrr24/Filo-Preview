from app import settings
import logging

logger = logging.getLogger(__name__)

states = settings.STATES
context_settings = settings.CONTEXT_SETTINGS
prompt_templates = settings.PROMPT_TEMPLATES

class ContextBuilder:
    def __init__(self, chat_vecstore, models):
        self.chat_vecstore = chat_vecstore
        self.decision = states["context_decision"][0]
        self.models = models

    def decide_context(self, query: str) -> str:
        """Decide wich type of context is needed"""
        msg = query.lower().strip()

        if len(msg) < 5 and any(word in msg for word in context_settings["short_context"]):
            self.decision = states["context_decision"][1]
            return self.decision
        
        if any(kw in msg for kw in context_settings["long_context"]):
            self.decision = states["context_decision"][2]
            return self.decision
        
        if msg in context_settings["no_context"]:
            self.decision = states["context_decision"][0]
            return self.decision
        
        # If it is not clear, use small llm to decide
        self.decision = self.models['small_llm'].invoke(
            prompt_templates["decide_context_prompt"].format(query=query)
        ).content.strip()

        return self.decision

    def build_context(self) -> str:
        """Use context methods depending of type of context"""
        if self.decision == states["context_decision"][0]:
            return ""
        
        elif self.decision == states["context_decision"][1]:
            self.chat_vecstore.get_buffer_memory
            
        elif self.decision == states["context_decision"][2]:
            return "\n".join([doc.page_content for doc in self.chat_vecstore.get_k_messages()])
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm
import sys
from ModelManager_streamlit import ModelManager
from langchain_core.output_parsers import StrOutputParser


class LLMQueryManager:
    """Manages LLM interactions and conversation history."""
    
    def __init__(
        self,
        llm_type: str,
        llm_model: str,
    ):
        """
        Initialize LLM interface.
        
        Args:
            llm_type: Type of LLM to use (e.g., 'OLLAMA')
            llm_model: Specific model to use
        """
        self.model_manager = ModelManager()
        self.parser = StrOutputParser()
        self.llm_type = llm_type
        self.llm_model = llm_model
        self.model = self._initialize_model()
        self.conversation_history: List[Dict[str, str]] = []

        # Configuration constants
        self.max_history_length = 50
        self.max_tokens_per_message = 1000
        self.max_context_tokens = 8000

    def _initialize_model(self) -> Any:
        """Initialize the language model."""
        config = {"selected_llm_type": self.llm_type}
        model, _, _, self.selected_llm = self.model_manager.validate_and_load_models(
            config=config,
            select_llm=self.llm_model,
            resource_manager={}
        )
        return model

    def add_to_history(self, role: str, content: str) -> None:
        """
        Add message to conversation history with token management.
        
        Args:
            role: Message role ('user' or 'assistant')
            content: Message content
        """
        # Trim history if exceeding limits
        while (len(self.conversation_history) >= self.max_history_length or
               self._estimate_total_tokens() >= self.max_context_tokens):
            self.conversation_history.pop(0)
            
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def _estimate_total_tokens(self) -> int:
        """Estimate total tokens in conversation history."""
        return sum(len(msg["content"].split()) * 1.3 
                  for msg in self.conversation_history)

    def get_conversation_context(self) -> str:
        """Get formatted conversation history."""
        return "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in self.conversation_history
        ])

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []

    def ask(self, question: str, use_history: bool = True) -> str:
        """
        Process a question and return the model's response.
        
        Args:
            question: User's question
            use_history: Whether to include conversation history
            
        Returns:
            str: Model's response
        """
        try:
            self.add_to_history("user", question)
            prompt = (
                self.get_conversation_context() if use_history and self.conversation_history 
                else question
            )

            with tqdm(total=100, desc="Generating response", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}', 
                     file=sys.stdout) as pbar:
                
                chain = self.model | self.parser
                pbar.update(30)
                response = chain.invoke(prompt)
                pbar.update(70)

            self.add_to_history("assistant", response)
            return response
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            print(f"\nError: {error_msg}")
            return error_msg

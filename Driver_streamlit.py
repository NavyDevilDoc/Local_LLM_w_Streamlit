from typing import List, Dict, Optional
from LLMQueryManager import LLMQueryManager


class Driver:
    """Driver class for LLM interaction and conversation management."""
    
    def __init__(
        self,
        llm_type: str = 'OLLAMA',
        llm_model: str = 'granite3.1-dense:8b-instruct-q4_0'
    ):
        """
        Initialize Driver with LLM configuration.
        
        Args:
            llm_type: Type of LLM to use (default: OLLAMA)
            llm_model: Specific model to use (default: granite3.1)
        """
        self.llm_type = llm_type
        self.llm_model = llm_model
        
        # Initialize LLM query manager
        self.llm_query = LLMQueryManager(
            llm_type=llm_type,
            llm_model=llm_model
        )

    def process_query(self, query: str, use_history: bool = True) -> str:
        """
        Process a user query and return the LLM response.
        
        Args:
            query: User's input query
            use_history: Whether to use conversation history
            
        Returns:
            str: LLM response
        """
        if not query.strip():
            return "Query cannot be empty."
        return self.llm_query.ask(query, use_history=use_history)

    def clear_conversation_history(self) -> None:
        """Reset the conversation history."""
        if self.llm_query:
            self.llm_query.conversation_history = []
            
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the current conversation history.
        
        Returns:
            List[Dict[str, str]]: List of conversation entries
        """
        return self.llm_query.conversation_history if self.llm_query else []
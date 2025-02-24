# Driver.py - Driver class for RAG model execution

import os
from LLMQueryManager import LLMQueryManager
import warnings
import json
from datetime import datetime
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Driver:
    def __init__(
        self,
        env_path: str,
        json_path: str,
        llm_type: str = 'OLLAMA',
        llm_model: str = 'llama3.2:latest', 
    ):
        self.env_path = env_path
        self.json_path = json_path
        self.llm_type = llm_type
        self.llm_model = llm_model

        # Initialize LLM query manager for direct queries
        self.llm_query = LLMQueryManager(
            env_path=env_path,
            json_path=json_path,
            llm_type=llm_type,
            llm_model=llm_model,
        )


    def process_query(self, query: str, use_history: bool = True) -> str:
        if not query.strip():
            return "Query cannot be empty."
        response = self.llm_query.ask(query, use_history=use_history)
        return response

    # Add this new method
    def get_model_info(self) -> dict:
        """Get information about the currently loaded model."""
        if self.llm_query:
            return self.llm_query.get_model_info()
        return {
            "type": "None",
            "model": "None",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def clear_conversation_history(self):
        """Clear the conversation history in LLM mode."""
        if self.llm_query:
            # Save current conversation history with a user-defined name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"conversation_history_{timestamp}.json"
            user_input = input(f"Enter a name for the conversation history file (default: {default_name}): ")
            user_named_file = os.path.join(self.llm_query.json_path, user_input or default_name)
            
            # Copy contents of conversation_history.json to user-named file
            if os.path.exists(self.llm_query.history_file):
                with open(self.llm_query.history_file, 'r', encoding='utf-8') as src:
                    conversation_history = json.load(src)
                with open(user_named_file, 'w', encoding='utf-8') as dst:
                    json.dump(conversation_history, dst, indent=2)
            else:
                print(f"Warning: {self.llm_query.history_file} does not exist. No history to copy.")
            
            # Clear the conversation history
            self.llm_query.conversation_history = []
            self.llm_query.save_history()
            
            return user_input or default_name  # Return the user-defined name


    def load_existing_conversation(self, file_path: str):
        """Load an existing conversation history from a specified file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.llm_query.conversation_history = json.load(f)
                self.llm_query.save_history()
        except Exception as e:
            print(f"Error loading existing conversation: {e}")


    def get_conversation_history(self):
        """Get the current conversation history in LLM mode."""
        if self.llm_query:
            return self.llm_query.conversation_history
        return []
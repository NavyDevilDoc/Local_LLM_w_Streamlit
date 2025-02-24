# LLMQueryManager.py
import os
import json
import sys
from tqdm import tqdm
from datetime import datetime
from typing import Any, Optional
from pathlib import Path
from ModelManager import ModelManager
from langchain_core.output_parsers import StrOutputParser


class LLMQueryManager:
    def __init__(
        self,
        env_path: str,
        json_path: str,
        llm_type: str,
        llm_model: str,
    ):
        """Initialize LLM interface."""
        self.model_manager = ModelManager(env_path)
        self.json_path = Path(json_path)
        self.parser = StrOutputParser()
        self.llm_type = llm_type
        self.llm_model = llm_model
        self.current_model_info = {
            "type": llm_type,
            "model": llm_model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.model = self._initialize_model()
        self.conversation_history = []
        self.history_file = self.json_path / "conversation_history.json"

        # Create directory and initialize history
        self.json_path.mkdir(parents=True, exist_ok=True)
        self._initialize_history_file()

        self.max_history_length = 50
        self.max_tokens_per_message = 1000
        self.max_context_tokens = 8000


    def _initialize_history_file(self):
        """Initialize history file if needed."""
        if not self.history_file.exists() or self.history_file.stat().st_size == 0:
            self.conversation_history = []
            self.save_history()
        else:
            self.conversation_history = self.load_history()
        

    def set_history_file(self, file_path: str):
        """Set a custom history file path."""
        self.history_file = file_path
        self.conversation_history = self.load_history()


    def _initialize_ollama_model_with_progress(self):
        """Initialize Ollama model"""
        self._initialize_model()


    def _initialize_model(self) -> Any:
        """Initialize the language model."""
        config = {
            "selected_llm_type": self.llm_type
        }
        model, _, _, self.selected_llm = self.model_manager.validate_and_load_models(
            config=config,
            select_llm=self.llm_model,
            resource_manager={}
        )
        # Add model info to session state
        self.current_model_info = self.model_manager.get_current_model_info()
        return model

    def get_model_info(self) -> dict:
        """Return current model information."""
        return self.current_model_info

    def load_history(self) -> list:
        """Load conversation history from JSON file."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading history: {e}")
        return []


    def save_history(self):
        """Save conversation history to JSON file."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")


    def add_to_history(self, role: str, content: str):
        """Add message with history management."""
        # Trim history if exceeding limits
        while (len(self.conversation_history) >= self.max_history_length or
               self._estimate_total_tokens() >= self.max_context_tokens):
            self.conversation_history.pop(0)  # Remove oldest message
        # Add new message
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.save_history()
        
        
    def _estimate_total_tokens(self) -> int:
        """Rough token count estimation."""
        return sum(len(msg["content"].split()) * 1.3 
                  for msg in self.conversation_history)
        

    def get_conversation_context(self) -> str:
        return "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in self.conversation_history
        ])


    def clear_history(self) -> Optional[str]:
        """Clear conversation history after backing up."""
        if not self.conversation_history:
            return None

        # Backup current history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.json_path / f"conversation_backup_{timestamp}.json"
        
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, indent=2)

        self.conversation_history = []
        self.save_history()
        return str(backup_file)


    def ask(self, question: str, use_history: bool = True) -> str:
        try:
            # Print formatted question
            print("\n" + "="*50)
            print("Question: " + question)
            print("="*50 + "\n")

            self.add_to_history("user", question)
            prompt = (
                self.get_conversation_context() if use_history and self.conversation_history 
                else question
            )
            # Create progress bar
            with tqdm(total=100, desc="Generating response", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}', 
                     file=sys.stdout) as pbar:
                
                # Initialize the chain
                chain = self.model | self.parser
                
                # Update progress to show model loaded
                pbar.update(30)
                
                # Get response
                response = chain.invoke(prompt)
                
                # Update progress to show completion
                pbar.update(70)

            self.add_to_history("assistant", response)
            return response
            
        except Exception as e:
            error_msg = f"Error processing question: {e}"
            print("\nError: " + "="*50)
            print(error_msg)
            print("="*50 + "\n")
            return error_msg
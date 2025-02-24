
from typing import List, Any, Dict, Tuple
import os
import datetime
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from ComputeResourceManager import ComputeResourceManager

class ModelManager:
    def __init__(self, env_path: str):
        """Initialize ModelManager with environment path and compute resources."""
        self.llm_type = None
        self.current_model = None
        self.compute_manager = ComputeResourceManager()
        self.load_environment_variables(env_path)

    def load_environment_variables(self, env_path: str) -> None:
        """Load environment variables from .env file."""
        try:
            print("Loading environment variables...")
            load_dotenv(env_path)
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            if not self.openai_api_key:
                print("Warning: OPENAI_API_KEY not found in environment variables")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load environment variables: {e}") from e

    def get_openai_api_key(self) -> str:
        """Retrieve the OpenAI API key"""
        return self.openai_api_key


    def validate_selection(self, selection: str, valid_choices: List[str]) -> None:
        """Validate model selection with case-insensitive matching."""
        selection_upper = selection.upper()
        valid_choices_upper = [choice.upper() for choice in valid_choices]
        if selection_upper not in valid_choices_upper:
            raise ValueError(
                f"Invalid selection: {selection}. "
                f"Available choices are: {valid_choices}"
            )


    def load_model(self, selected_llm_type: str, selected_llm: str, resource_manager: Dict) -> Any:
        try:
            self.llm_type = selected_llm_type.upper()
            self.current_model = selected_llm
            self.compute_manager.test_gpu_details()
            
            # Get optimal compute settings
            compute_settings = self.compute_manager.get_compute_settings()
            
            if selected_llm_type.lower() in ["gpt", "openai"]:
                if not hasattr(self, 'openai_api_key'):
                    raise ValueError("OpenAI API key not initialized. Did you load environment variables?")
                if not self.openai_api_key:
                    raise ValueError("OpenAI API key is empty. Please check your .env file.")
                    
                model = ChatOpenAI(
                    model=selected_llm,  # Add this line to specify the model
                    openai_api_key=self.openai_api_key, 
                    temperature=compute_settings['temperature'],
                    streaming=True
                )
                print(f"Loaded OpenAI model: {selected_llm}")
                return model
            
            elif selected_llm_type.lower() == "ollama":
                model =  ChatOllama(
                    model=selected_llm, 
                    top_k=compute_settings['top_k'],
                    top_p=compute_settings['top_p'],
                    temperature=compute_settings['temperature'],
                    disable_streaming=False
                )
                print(f"Loaded Ollama model: {selected_llm}")
                return model
            else:
                raise ValueError(f"Unsupported LLM type: {selected_llm_type}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model: {e}") from e

    def get_current_model_info(self) -> dict:
        """Return information about the currently loaded model."""
        return {
            "type": self.llm_type,
            "model": self.current_model,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def validate_and_load_models(
        self, 
        config: Dict[str, str], 
        select_llm: str, 
        resource_manager: Any
    ) -> Tuple[Any, str, None, str]:  # Update return type hint
        try:
            normalized_config = {k.upper(): v.upper() for k, v in config.items()}
            required_keys = ["SELECTED_LLM_TYPE"]
            # Validate config
            missing_keys = [k for k in required_keys if k not in normalized_config]
            if missing_keys:
                raise KeyError(f"Missing required config keys: {missing_keys}")
            
            # Load and validate models
            model = self.load_model(
                normalized_config["SELECTED_LLM_TYPE"],
                select_llm,
                resource_manager
            )
            # Return tuple matching expected format
            return model, None, None, select_llm  # Match the expected 4 values
            
        except Exception as e:
            print(f"Model validation failed: {e}")
            raise
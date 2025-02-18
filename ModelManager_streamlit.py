from typing import List, Any, Dict, Tuple
from langchain_ollama import ChatOllama
from ComputeResourceManager_streamlit import ComputeResourceManager


class ModelManager:
    """Manages LLM model initialization and configuration."""
    
    def __init__(self):
        """Initialize ModelManager with compute resources."""
        self.llm_type = None
        self.compute_manager = ComputeResourceManager()

    def validate_selection(self, selection: str, valid_choices: List[str]) -> None:
        """
        Validate model selection with case-insensitive matching.
        
        Args:
            selection: Model selection string
            valid_choices: List of valid model choices
            
        Raises:
            ValueError: If selection is not in valid_choices
        """
        selection_upper = selection.upper()
        valid_choices_upper = [choice.upper() for choice in valid_choices]
        if selection_upper not in valid_choices_upper:
            raise ValueError(
                f"Invalid selection: {selection}. "
                f"Available choices are: {valid_choices}"
            )

    def load_model(self, selected_llm_type: str, selected_llm: str, 
                  resource_manager: Dict) -> Any:
        """
        Load and configure the selected LLM model.
        
        Args:
            selected_llm_type: Type of LLM (must be 'ollama')
            selected_llm: Specific model name
            resource_manager: Resource configuration dictionary
            
        Returns:
            ChatOllama: Configured Ollama chat model
            
        Raises:
            ValueError: If unsupported LLM type is specified
        """
        try:
            self.llm_type = selected_llm_type.upper()
            ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            compute_settings = self.compute_manager.get_compute_settings()
            
            if selected_llm_type.lower() != "ollama":
                raise ValueError("Only Ollama models are supported")
                
            return ChatOllama(
                model=selected_llm,
                base_url = ollama_base_url,
                top_k=compute_settings['top_k'],
                top_p=compute_settings['top_p'],
                temperature=compute_settings['temperature'],
                disable_streaming=False
            )
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model: {e}") from e

    def validate_and_load_models(
        self, 
        config: Dict[str, str], 
        select_llm: str, 
        resource_manager: Any
    ) -> Tuple[Any, None, None, str]:
        """
        Validate configuration and load models.
        
        Args:
            config: Configuration dictionary
            select_llm: Selected model name
            resource_manager: Resource manager instance
            
        Returns:
            Tuple containing (model, None, None, model_name)
        """
        try:
            normalized_config = {k.upper(): v.upper() for k, v in config.items()}
            if "SELECTED_LLM_TYPE" not in normalized_config:
                raise KeyError("Missing required config key: SELECTED_LLM_TYPE")
            
            model = self.load_model(
                normalized_config["SELECTED_LLM_TYPE"],
                select_llm,
                resource_manager
            )
            return model, None, None, select_llm
            
        except Exception as e:
            print(f"Model validation failed: {e}")
            raise

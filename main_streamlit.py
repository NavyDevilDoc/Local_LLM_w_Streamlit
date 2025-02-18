from pathlib import Path
from Setup_streamlit import initiate_llm
from Driver_streamlit import Driver_streamlit

def main() -> None:
    """Initialize and run the LLM interface with Ollama."""
    # Create a default path for conversation storage in the app directory
    conversations_path = Path("conversations")
    conversations_path.mkdir(exist_ok=True)

    driver = Driver(
        env_path="",  # No env file needed for Ollama
        json_path=str(conversations_path),
        llm_type="ollama",
        llm_model="granite3.1-dense:8b-instruct-q4_0"
    )

    initiate_llm(driver)

if __name__ == "__main__":
    main()

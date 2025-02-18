from Driver_streamlit import Driver
from TextPreprocessor import TextPreprocessor

def initiate_llm(driver: Driver) -> None:
    """
    Initialize LLM interface and manage chat session.
    
    Args:
        driver: Driver instance for LLM interaction
    """
    text_preprocessor = TextPreprocessor()
    
    # Start fresh conversation
    driver.clear_conversation_history()
    print("\nStarting new chat session...")
    
    # Main chat loop
    while True:
        user_input = input("\nEnter your query (or type 'exit/quit' to quit): ")
        if user_input.lower() in ['exit', 'quit']:
            print("Ending chat session.")
            break
            
        response = driver.process_query(user_input)
        
        # Format and display response
        print("\n" + "="*50)
        print("Answer:")
        print("="*50)
        print(text_preprocessor.format_text(response))
        print("="*50)

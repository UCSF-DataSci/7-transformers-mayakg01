import requests
import argparse
import os

def get_response(prompt, model_name, api_key=None):
    """
    Get a response from the model
    
    Args:
        prompt: The prompt to send to the model
        model_name: Name of the model to use
        api_key: API key for authentication (optional for some models)
        
    Returns:
        The model's response
    """
    # Implement the get_response function
    # Set up the API URL and headers
    # Create a payload with the prompt
    # Send the payload to the API
    # Extract and return the generated text from the response
    # Handle any errors that might occur
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    api_token = api_key
    if not api_token:
        raise ValueError("Hugging Face API token not provided")
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 100}
    }
    print(f"Requesting from: {API_URL}")
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    if isinstance(result, list) and "generated_text" in result[0]:
        return result[0]["generated_text"]
    elif "error" in result:
        return f"Error: {result['error']}"
    else:
        return "Unexpected response format."
    
def run_chat(model_name, api_key):
    """Run an interactive chat session"""
    print("Welcome to the Simple LLM Chat! Type 'exit' to quit.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
            
        # Get response from the model
        response = get_response(user_input, model_name=model_name, api_key=api_key)
        # Print the response
        print("LLM:", response)
def main():
    parser = argparse.ArgumentParser(description="Chat with an LLM")
    #Add arguments to the parser
    parser = argparse.ArgumentParser(description="Chat with an LLM")
    parser.add_argument("--model", type=str, default="google/flan-t5-base", help="Hugging Face Model to be used.")
    parser.add_argument("--api_key", type=str, help="API key from Hugging Face.")
    args = parser.parse_args()
    
    # Run the chat function with parsed arguments
    run_chat(model_name=args.model, api_key=args.api_key)
if __name__ == "__main__":
    main()
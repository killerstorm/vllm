import requests
import json
import time
import sys

# --- Configuration ---
# Replace with your vLLM server details and model name
VLLM_HOST = "localhost"
VLLM_PORT = 8000  # Default vLLM OpenAI-compatible server port
BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"  # IMPORTANT: Change to a model loaded in your vLLM server
MODEL_NAME = "Qwen/Qwen3-0.6B"

# --- Helper Functions (from verified_endpoints_example.py) ---
def print_json(data, title="Response"):
    """Prints JSON data with a title, formatted for readability."""
    print(f"--- {title} ---")
    try:
        print(json.dumps(data, indent=2))
    except TypeError:
        print(data) # Print as is if not serializable to JSON directly
    print("-" * (len(title) + 8) + "\n")

def check_token_details(token_details, type_name="Tokens"):
    """
    Prints details for a list of tokens and checks if all are greedy (rank 1).
    Returns True if all tokens with a rank are greedy, False otherwise.
    """
    if not token_details:
        print(f"  ({type_name} details not available or empty)")
        return True # Considered "greedy" if no details to check or not applicable

    all_greedy = True
    print(f"  {type_name} Details:")
    for td in token_details:
        token_id = td.get('token_id', 'N/A')
        text = td.get('text', '')
        logprob = td.get('logprob')
        rank = td.get('rank')
        is_greedy_choice = td.get('is_greedy_choice')
        top_logprob_at_step = td.get('top_logprob_at_step')
        top_token_id_at_step = td.get('top_token_id_at_step')
        error_message = td.get('error_message')
        
        print(f"    ID: {str(token_id):<5} Text: '{text}' Logprob: {logprob:<8.4f} Rank: {str(rank):<3} Greedy: {is_greedy_choice}")
        if rank != 1:
            print(f"    Logprob: {logprob} vs {top_logprob_at_step}")
    return all_greedy

def test_long_chat_and_verify():
    """
    Tests generation of a longer sequence via chat completion and then verifies it.
    Reports detailed data only if the test fails.
    """
    print("=== Testing Long Chat Completion & Verification ===")
    chat_endpoint = f"{BASE_URL}/chat/completions/verified"
    verify_endpoint = f"{BASE_URL}/verify_decoding"

    story_prompt = "Write a short story about a curious robot exploring a mysterious ancient library. The story should be at least 100 words long."
    max_story_tokens = 250  # Allow for a decent length story

    chat_payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a creative and skilled storyteller."},
            {"role": "user", "content": story_prompt}
        ],
        "max_tokens": max_story_tokens,
        "temperature": 0.0, # Ensure greedy generation for verification
        "prompt_logprobs": 1, # Essential to get prompt_token_details with token_ids
        "logprobs": True,     # To get completion_token_details if needed for debugging
        "top_logprobs": 1   # Ensures logprobs data is rich for completion tokens
    }

    print(f"Requesting chat completion for a story (max_tokens: {max_story_tokens})...")
    try:
        # 1. Generate the story via chat completions
        chat_response = requests.post(chat_endpoint, json=chat_payload)
        chat_response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        chat_data = chat_response.json()

        if not chat_data.get("choices"):
            print("ERROR: No choices in chat completion response.")
            print_json(chat_data, "Chat Completion Error Response")
            sys.exit(1)
        
        choice = chat_data["choices"][0]
        generated_story = choice.get("message", {}).get("content")
        
        if not generated_story:
            print("ERROR: No story content generated.")
            print_json(chat_data, "Chat Completion Error Response")
            sys.exit(1)
        
        prompt_token_details = choice.get("prompt_token_details")
        if not prompt_token_details:
            print("ERROR: Prompt token details not found in chat completion response. Cannot proceed with /verify_decoding.")
            print_json(chat_data, "Chat Completion Error Response")
            sys.exit(1)

        prompt_token_ids = [detail['token_id'] for detail in prompt_token_details if 'token_id' in detail]
        if not prompt_token_ids:
            print("ERROR: Extracted prompt token IDs are empty or malformed.")
            print_json(chat_data, "Chat Completion Error Response")
            sys.exit(1)

        print(f"Story generated (approx. {len(generated_story.split())} words). Verifying greedy decoding...")

        # 2. Verify the generated story
        verify_payload = {
            "model": MODEL_NAME,
            "prompt": prompt_token_ids,      # Use token IDs from the chat response
            "completion": generated_story,   # The actual text of the story
            "check_greedy": True
        }

        verify_response = requests.post(verify_endpoint, json=verify_payload)
        verify_response.raise_for_status()
        verify_data = verify_response.json()

        if verify_data.get("is_verified_greedy") is True:
            print("SUCCESS: Long chat sequence verified as greedy.")
        else:
            print("\nFAILURE: Long chat sequence DID NOT verify as greedy.")
            print("--- Failing Test Details ---")
            print("\n--- Chat Completion Request ---")
            print(json.dumps(chat_payload, indent=2))
            print("\n--- Chat Completion Response ---")
            print(json.dumps(chat_data, indent=2))
            print("\n--- Verification Request ---")
            print(json.dumps(verify_payload, indent=2))
            print("\n--- Verification Response ---")
            print(json.dumps(verify_data, indent=2))
            
            if verify_data.get("verification_details"):
                print("\n--- Verification Token Details (from /verify_decoding) ---")
                check_token_details(verify_data.get("verification_details"), "Verification")
            sys.exit(1)

    except requests.exceptions.RequestException as e:
        print(f"\nERROR: API Request failed: {e}")
        if e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response content: {e.response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)
    print("="*55 + "\n")


if __name__ == "__main__":
    print(f"--- Starting vLLM Long Sequence Chat & Verification Test ---")
    print(f"Targeting server: {BASE_URL}")
    print(f"Using model: {MODEL_NAME}")
    print("Important: Ensure this model is loaded in your vLLM server, and the server supports the /verified endpoints.")
    print(f"  Example server command: python -m vllm.entrypoints.openai.api_server --model {MODEL_NAME}")
    print("-" * 60 + "\n")

    test_long_chat_and_verify()

    print("--- Test finished ---") 

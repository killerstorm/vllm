import requests
import json
import time
import sys # Added for sys.exit

# --- Configuration ---
# Replace with your vLLM server details and model name
VLLM_HOST = "localhost"
VLLM_PORT = 8000  # Default vLLM OpenAI-compatible server port
BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"  # IMPORTANT: Change to a model loaded in your vLLM server

# --- Helper Functions ---
def print_json(data, title="Response"):
    print(f"--- {title} ---")
    try:
        print(json.dumps(data, indent=2))
    except TypeError:
        print(data) # Print as is if not serializable to JSON directly
    print("-" * (len(title) + 8) + "\n")

def check_token_details(token_details, type_name="Tokens"):
    if not token_details:
        print(f"  ({type_name} details not available or empty)")
        return True # Or False depending on whether they are expected

    all_greedy = True
    print(f"  {type_name} Details:")
    for td in token_details:
        token_id = td.get('token_id', 'N/A')
        text = td.get('text', '')
        logprob = td.get('logprob', 0.0)
        rank = td.get('rank')
        is_greedy_char = '✔' if rank == 1 else ('✘' if rank is not None else '?')
        print(f"    ID: {token_id:<5} Text: '{text}' Logprob: {logprob:<8.4f} Rank: {str(rank):<3} Greedy: {is_greedy_char}")
        if rank is not None and rank != 1:
            all_greedy = False
    return all_greedy

# --- Test Functions ---

def test_verified_completion():
    print("=== Testing /v1/completions/verified ===")
    endpoint = f"{BASE_URL}/completions/verified"
    payload = {
        "model": MODEL_NAME,
        "prompt": "San Francisco is a city in",
        "max_tokens": 10,
        "prompt_logprobs": 5, # Request logprobs for prompt tokens
        "logprobs": 5         # Request logprobs (and rank) for completion tokens (maps to top_logprobs for engine)
    }
    print(f"Requesting: {endpoint} with payload:\n{json.dumps(payload, indent=2)}")

    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        data = response.json()
        print_json(data, "Verified Completion Response")

        if not data.get("choices"):
            print("ERROR: No choices in response.")
            sys.exit(1) # Exit on error

        choice = data["choices"][0]
        
        print("\n--- Sanity Checks & Visualization (Verified Completion) ---")
        print(f"Generated Text: {choice.get('text')}")
        
        print("\nPrompt Token Details:")
        check_token_details(choice.get("prompt_token_details"), "Prompt")
        
        print("\nCompletion Token Details:")
        all_completion_greedy = check_token_details(choice.get("completion_token_details"), "Completion")
        
        if not all_completion_greedy:
            print("WARNING: Not all completion tokens were rank 1. This is unexpected for temperature=0.")
        else:
            print("SUCCESS: All completion tokens appear to be greedy (rank 1).")

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Request failed: {e}")
        if e.response is not None:
            print(f"Response content: {e.response.text}")
        sys.exit(1) # Exit on error
    print("="*40 + "\n")


def test_verified_chat_completion():
    print("=== Testing /v1/chat/completions/verified ===")
    endpoint = f"{BASE_URL}/chat/completions/verified"
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "What is the capital of California?"}
        ],
        "max_tokens": 15,
        "prompt_logprobs": 3,
        "logprobs": True,  # Explicitly set logprobs to True
        "top_logprobs": 3 # This ensures sampling_params.logprobs is set for completion
    }
    print(f"Requesting: {endpoint} with payload:\n{json.dumps(payload, indent=2)}")

    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        data = response.json()
        print_json(data, "Verified Chat Completion Response")

        if not data.get("choices"):
            print("ERROR: No choices in response.")
            sys.exit(1) # Exit on error

        choice = data["choices"][0]
        message = choice.get("message", {})
        
        print("\n--- Sanity Checks & Visualization (Verified Chat Completion) ---")
        print(f"Assistant's Message: {message.get('content')}")

        print("\nPrompt Token Details (Full History):")
        check_token_details(choice.get("prompt_token_details"), "Prompt")

        print("\nCompletion Token Details (Assistant's Message):")
        all_completion_greedy = check_token_details(choice.get("completion_token_details"), "Completion")

        if not all_completion_greedy:
            print("WARNING: Not all completion tokens were rank 1. This is unexpected for temperature=0.")
        else:
            print("SUCCESS: All completion tokens appear to be greedy (rank 1).")

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Request failed: {e}")
        if e.response is not None:
            print(f"Response content: {e.response.text}")
        sys.exit(1) # Exit on error
    print("="*40 + "\n")


def test_verify_decoding():
    print("=== Testing /v1/verify_decoding ===")
    endpoint = f"{BASE_URL}/verify_decoding"

    # Case 1: Known/expected greedy sequence (using text)
    prompt1 = "The first prime numbers are"
    # Assuming a model might complete this as " 2, 3, 5, 7" (note leading space)
    # For a robust test, this completion should be one actually generated by the target model greedily.
    completion1_text = " 2, 3, 5, 7" 
    payload1 = {
        "model": MODEL_NAME,
        "prompt": prompt1,
        "completion": completion1_text,
        "check_greedy": True,
        "greedy_logprob_threshold": 0.001 
    }
    print(f"Requesting (Case 1 - Text Input): {endpoint} with payload:\n{json.dumps(payload1, indent=2)}")
    try:
        response1 = requests.post(endpoint, json=payload1)
        response1.raise_for_status()
        data1 = response1.json()
        print_json(data1, "Verify Decoding Response (Case 1)")

        print("\n--- Sanity Checks & Visualization (Verify Decoding - Case 1) ---")
        print(f"Prompt Tokens: {data1.get('prompt_token_ids')}")
        print(f"Completion Tokens: {data1.get('completion_token_ids')}")
        print(f"Overall Greedy Verification: {data1.get('is_verified_greedy')}")
        if data1.get("is_verified_greedy") is not True:
             print("WARNING: Case 1 did not verify as greedy as expected.")
        else:
             print("SUCCESS: Case 1 verified as greedy.")
        check_token_details(data1.get("verification_details"), "Verification")
        
    except requests.exceptions.RequestException as e:
        print(f"ERROR (Case 1): Request failed: {e}")
        if e.response is not None:
            print(f"Response content: {e.response.text}")
        sys.exit(1) # Exit on error

    # Case 2: Using token IDs and a deliberately non-greedy token (if known)
    # This requires knowing actual token IDs and non-greedy choices for your model.
    # For now, let's re-verify the previous completion but with token IDs.
    # First, we need to get the token IDs for prompt1 and completion1_text
    try:
        print("\n--- Preparing for Case 2: Tokenizing prompt and completion from Case 1 ---")
        # Corrected tokenize_endpoint to not use /v1 prefix
        tokenize_endpoint = f"http://{VLLM_HOST}:{VLLM_PORT}/tokenize" 
        prompt_tokens_resp_req = requests.post(tokenize_endpoint, json={"model": MODEL_NAME, "prompt": prompt1, "add_special_tokens": False})
        prompt_tokens_resp_req.raise_for_status()
        prompt_tokens_resp = prompt_tokens_resp_req.json()

        completion_tokens_resp_req = requests.post(tokenize_endpoint, json={"model": MODEL_NAME, "prompt": completion1_text, "add_special_tokens": False})
        completion_tokens_resp_req.raise_for_status()
        completion_tokens_resp = completion_tokens_resp_req.json()
        
        prompt1_ids = prompt_tokens_resp.get("tokens")
        completion1_ids = completion_tokens_resp.get("tokens")

        if not prompt1_ids or not completion1_ids:
            print("ERROR: Could not tokenize prompt/completion for Case 2. Skipping.")
            # We might want to exit here too if tokenization is critical for further tests
            # For now, it just skips this part of Case 2 as per original logic.
            # If it should be a hard fail, add sys.exit(1)
            return # Original behavior was to return from function
        
        print(f"Tokenized prompt1: {prompt1_ids}")
        print(f"Tokenized completion1_text: {completion1_ids}")

        payload2 = {
            "model": MODEL_NAME,
            "prompt": prompt1_ids,
            "completion": completion1_ids,
            "check_greedy": True
        }
        print(f"\nRequesting (Case 2 - Token ID Input): {endpoint} with payload:\n{json.dumps(payload2, indent=2)}")
        response2 = requests.post(endpoint, json=payload2)
        response2.raise_for_status()
        data2 = response2.json()
        print_json(data2, "Verify Decoding Response (Case 2 - Token IDs)")
        print("\n--- Sanity Checks & Visualization (Verify Decoding - Case 2) ---")
        print(f"Overall Greedy Verification: {data2.get('is_verified_greedy')}")
        if data2.get("is_verified_greedy") is not True:
             print("WARNING: Case 2 (token ID input) did not verify as greedy.")
        else:
             print("SUCCESS: Case 2 (token ID input) verified as greedy.")
        check_token_details(data2.get("verification_details"), "Verification")

        # Case 2b: Make the last token non-greedy (example, assumes ID 100 is valid but non-greedy here)
        # This is a placeholder. A robust test needs actual non-greedy token knowledge.
        if completion1_ids:
            modified_completion_ids = list(completion1_ids)
            original_last_token = modified_completion_ids[-1]
            # Try to pick a different token. This is a naive attempt.
            # A better way would be to query possible tokens. For now, just an example.
            modified_completion_ids[-1] = original_last_token + 1 if original_last_token < 50000 else original_last_token -1 
            
            payload2b = {
                "model": MODEL_NAME,
                "prompt": prompt1_ids,
                "completion": modified_completion_ids,
                "check_greedy": True
            }
            print(f"\nRequesting (Case 2b - Modified Token ID): {endpoint} with payload:\n{json.dumps(payload2b, indent=2)}")
            response2b = requests.post(endpoint, json=payload2b)
            # We expect this might fail or show non-greedy if the token is valid but not greedy
            # or fail if the token is invalid.
            if response2b.status_code == 200:
                data2b = response2b.json()
                print_json(data2b, "Verify Decoding Response (Case 2b - Modified)")
                print(f"Overall Greedy Verification (Modified): {data2b.get('is_verified_greedy')}")
                if data2b.get("is_verified_greedy") is True:
                    print("WARNING: Case 2b (modified) unexpectedly verified as greedy. Modification might have been lucky or token invalid leading to odd behavior.")
                else:
                    print("SUCCESS (expected): Case 2b (modified) did NOT verify as greedy or failed as expected.")
                check_token_details(data2b.get("verification_details"), "Verification (Modified)")
            else:
                print(f"INFO (Case 2b - Modified): Request returned status {response2b.status_code}. This might be expected if modification led to an error.")
                print(f"Response content: {response2b.text}")


    except requests.exceptions.RequestException as e:
        print(f"ERROR (Case 2/2b Preparations or Execution): Request failed: {e}") # Combined error message
        if e.response is not None:
            print(f"Response content: {e.response.text}")
        sys.exit(1) # Exit on error
            
    print("="*40 + "\n")


if __name__ == "__main__":
    print(f"--- Starting vLLM Verified Endpoint Tests ---")
    print(f"Targeting server: {BASE_URL}")
    print(f"Using model: {MODEL_NAME} (Please ensure this model is loaded in your vLLM server)")
    print("Make sure your vLLM server is running, e.g.:")
    print(f"  python -m vllm.entrypoints.openai.api_server --model {MODEL_NAME}")
    print("-" * 50 + "\n")

    test_verified_completion()
    time.sleep(0.5) # Small delay between tests

    test_verified_chat_completion()
    time.sleep(0.5)

    test_verify_decoding()

    print("--- All tests finished ---") 

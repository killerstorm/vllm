"""
This script tests the vLLM OpenAI-compatible server's extended endpoints:
- /v1/chat/completions/verified
- /v1/verify_decoding

It assumes the vLLM server is running and accessible at the specified URL,
and that the specified model is loaded.

Example Usage:
python examples/online_serving/test_openai_extended_api.py --model facebook/opt-125m
"""
import argparse
import json
import requests

# Default server URL and model.
# These might need to be changed depending on how the server is run.
DEFAULT_SERVER_URL = "http://localhost:8000/v1"
# Using a common small model, ensure this model is served by your vLLM instance
DEFAULT_MODEL_NAME = "facebook/opt-125m"


def make_request(url, method="POST", payload=None):
    """Helper function to make HTTP requests to the server."""
    headers = {"Content-Type": "application/json"}
    try:
        if method.upper() == "POST":
            response = requests.post(url, headers=headers, json=payload)
        elif method.upper() == "GET":
            response = requests.get(url, headers=headers)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making request to {url}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response content: {e.response.text}")
        return None


def test_chat_verified(server_url, model_name, prompt):
    """Tests the /v1/chat/completions/verified endpoint."""
    print(f"\n--- Testing /chat/completions/verified with prompt: '{prompt}' ---")
    endpoint_url = f"{server_url}/chat/completions/verified"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 50
    }
    response_data = make_request(endpoint_url, payload=payload)

    if not response_data:
        print("Test /chat/completions/verified FAILED: No response or error during request.")
        return None

    try:
        assert "id" in response_data, "Response missing 'id'"
        assert "choices" in response_data and len(response_data["choices"]) > 0, \
            "Response missing 'choices' or empty"
        choice = response_data["choices"][0]
        assert "message" in choice and "content" in choice["message"], \
            "Choice missing 'message.content'"

        assert isinstance(choice.get("prompt_token_ids"), list), "Choice missing 'prompt_token_ids' list"
        assert isinstance(choice.get("completion_token_ids"), list), "Choice missing 'completion_token_ids' list"
        assert isinstance(choice.get("completion_token_details"), list), "Choice missing 'completion_token_details' list"

        print("Test /chat/completions/verified PASSED.")
        print(f"  Prompt: {prompt}")
        print(f"  Response content: {choice['message']['content']}")
        print(f"  Prompt token IDs (first 10): {str(choice['prompt_token_ids'][:10])}...")
        print(f"  Completion token IDs (first 10): {str(choice['completion_token_ids'][:10])}...")
        return response_data
    except AssertionError as e:
        print(f"Test /chat/completions/verified FAILED: Assertion Error - {e}")
        print("Response data received:")
        print(json.dumps(response_data, indent=2))
        return None


def test_greedy_verification(server_url, model_name, prompt_for_greedy_source):
    """Tests the /v1/verify_decoding endpoint."""
    print(f"\n--- Testing /verify_decoding with base prompt: '{prompt_for_greedy_source}' ---")

    # Step 0: Get a greedy completion from /chat/tokens
    print("\nFetching initial greedy completion from /chat/completions/verified...")
    # Use a different prompt variable for clarity if needed, but reusing is fine
    chat_response = test_chat_verified(server_url, model_name, prompt_for_greedy_source)

    if not chat_response or not chat_response.get("choices"):
        print("Skipping /verify_decoding tests as /chat/completions/verified failed or returned invalid response.")
        return

    prompt_text = prompt_for_greedy_source # Assuming simple string prompt
    greedy_completion_text = chat_response["choices"][0]["message"]["content"]
    choice = chat_response["choices"][0]
    prompt_ids = choice["prompt_token_ids"]
    greedy_completion_ids = choice["completion_token_ids"]

    endpoint_url = f"{server_url}/verify_decoding"
    all_tests_passed = True

    def run_verification_test(test_name, payload, expect_verified, expect_failure_idx_state=None):
        nonlocal all_tests_passed
        print(f"\n{test_name}")
        response = make_request(endpoint_url, payload=payload)
        if not response:
            print(f"  Test FAILED: No response from /verify_decoding.")
            all_tests_passed = False
            return

        try:
            assert response.get("is_verified_greedy") is expect_verified, \
                f"Expected is_verified_greedy={expect_verified}, got {response.get('is_verified_greedy')}"
            
            print(f"  Test PASSED. is_verified_greedy: {response.get('is_verified_greedy')}")
                # print(f"    Expected Tokens (first 5): {logprobs_info.get('expected_tokens', [])[:5]}")
                # print(f"    Actual Tokens (first 5): {logprobs_info.get('tokens', [])[:5]}")

        except AssertionError as e:
            print(f"  Test FAILED: Assertion Error - {e}")
            print("  Payload sent:")
            print(json.dumps(payload, indent=2))
            print("  Response received:")
            print(json.dumps(response, indent=2))
            all_tests_passed = False
            
    # Test 1: Correct greedy completion using token IDs
    payload1 = {"model": model_name, "prompt": prompt_ids, "completion": greedy_completion_ids}
    run_verification_test("Test 1: Verifying correct greedy completion (using token IDs)", payload1, True)

    # Test 2: Correct greedy completion using text
    payload2 = {"model": model_name, "prompt": prompt_text, "completion": greedy_completion_text}
    run_verification_test("Test 2: Verifying correct greedy completion (using text)", payload2, True)

    # Test 3: Non-greedy completion (modified text)
    # Only run if we have something to make non-greedy.
    # If greedy_completion_text is empty, prepending disruptive_prefix makes it non-empty,
    # which should be non-greedy compared to an empty original.
    if greedy_completion_text or not greedy_completion_ids or prompt_text: # ensure there's context
        disruptive_prefix = " ZYXWVU " 
        non_greedy_completion_text = disruptive_prefix + greedy_completion_text
        
        payload3 = {"model": model_name, "prompt": prompt_text, "completion": non_greedy_completion_text}
        run_verification_test("Test 3: Verifying non-greedy completion (modified text)", 
                              payload3, False, expect_failure_idx_state="is_not_none")
    else:
        print("\nTest 3: Verifying non-greedy completion (modified text) SKIPPED (original completion context was insufficient).")


    # Test 4: Empty completion (using text) - should be trivially verified
    payload4 = {"model": model_name, "prompt": prompt_text, "completion": ""}
    run_verification_test("Test 4: Verifying empty completion (using text)", payload4, True)

    # Test 5: Empty completion (using token IDs) - should be trivially verified
    payload5 = {"model": model_name, "prompt_token_ids": prompt_ids, "completion_token_ids": []}
    run_verification_test("Test 5: Verifying empty completion (using token IDs)", payload5, True)

    if all_tests_passed:
        print("\nAll /verify_decoding sub-tests PASSED.")
    else:
        print("\nSome /verify_decoding sub-tests FAILED.")


def main():
    parser = argparse.ArgumentParser(
        description="Test OpenAI-compatible server's extended verified endpoints."
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=DEFAULT_SERVER_URL,
        help="Base URL of the vLLM OpenAI-compatible server (e.g., http://localhost:8000/v1).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Name of the model to use for testing (must be loaded on the server).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, world! My name is",
        help="Main prompt to use for testing."
    )
    
    args = parser.parse_args()

    print(f"Using server URL: {args.server_url}")
    print(f"Using model: {args.model}")

    # Test /chat/completions/verified with the main prompt
    test_chat_verified(args.server_url, args.model, args.prompt)
    
    # Test /verify/greedy using the main prompt as a base
    test_greedy_verification(args.server_url, args.model, args.prompt)

    # Test with a very short prompt as well to cover edge cases
    short_prompt = "A B C"
    print(f"\n--- Running tests with short prompt: '{short_prompt}' ---")
    test_chat_verified(args.server_url, args.model, short_prompt)
    test_greedy_verification(args.server_url, args.model, short_prompt)
    
    print("\n--- All tests finished ---")

if __name__ == "__main__":
    main() 
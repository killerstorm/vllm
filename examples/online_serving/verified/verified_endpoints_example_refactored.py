import sys
import time

from verification_utils import VerificationClient


def main():
    # Configuration
    VLLM_HOST = "34.70.189.211"
    VLLM_PORT = 5000
    MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"  # Change to your loaded model
    
    print(f"--- Starting vLLM Verified Endpoint Tests ---")
    print(f"Targeting server: http://{VLLM_HOST}:{VLLM_PORT}/v1")
    print(f"Using model: {MODEL_NAME}")
    print("Make sure your vLLM server is running, e.g.:")
    print(f"  python -m vllm.entrypoints.openai.api_server --model {MODEL_NAME}")
    print("-" * 50 + "\n")
    
    # Initialize client
    client = VerificationClient(VLLM_HOST, VLLM_PORT)
    
    # Test 1: Verified Completion
    #success = client.verify_completion(
    #    model=MODEL_NAME,
    #    prompt="San Francisco is a city in",
    #    max_tokens=10,
    #    prompt_logprobs=5
    #)
    #if not success:
    #    sys.exit(1)
    
    time.sleep(0.5)
    print("="*40 + "\n")
    
    # Test 2: Verified Chat Completion
    success = client.verify_chat_completion(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "What is the capital of California?"}
        ],
        max_tokens=15,
        prompt_logprobs=3
    )
    if not success:
        sys.exit(1)
    
    time.sleep(0.5)
    print("="*40 + "\n")
    
    # Test 3: Verify Decoding with text
    result = client.verify_decoding(
        model=MODEL_NAME,
        prompt="The first prime numbers are",
        completion=" 2, 3, 5, 7",
        check_greedy=True,
        greedy_logprob_threshold=0.001
    )
    if result is None:
        sys.exit(1)
    elif not result:
        print("WARNING: Completion did not verify as greedy.")
    else:
        print("SUCCESS: Completion verified as greedy.")
    
    time.sleep(0.5)
    print("="*40 + "\n")
    
    # Test 4: Batch Verification
    summary = client.batch_verify_decoding(
        model=MODEL_NAME,
        verification_pairs=[
            {"prompt": "The capital of France is", "completion": " Paris"},
            {"prompt": "2 + 2 equals", "completion": " 4"},
            {"prompt": "The sky is usually", "completion": " blue"},
            {"prompt": "Water freezes at", "completion": " 0 degrees Celsius"}
        ]
    )
    
    if summary.get("error"):
        print(f"ERROR: Batch verification failed: {summary['error']}")
        sys.exit(1)
    
    print(f"\nBatch verification completed successfully!")
    print(f"Total verified: {summary.get('verified', 0)}/{summary.get('total', 0)}")
    
    print("\n--- All tests finished successfully! ---")


if __name__ == "__main__":
    main()

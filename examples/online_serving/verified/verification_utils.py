"""Shared utilities for verification endpoint examples."""
import json
import sys
from typing import Any, Dict, List, Optional, Union

import requests


class VerificationClient:
    """Client for interacting with vLLM verification endpoints."""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}/v1"
        
    def print_json(self, data: Any, title: str = "Response") -> None:
        """Prints JSON data with a title, formatted for readability."""
        print(f"--- {title} ---")
        try:
            print(json.dumps(data, indent=2))
        except TypeError:
            print(data)
        print("-" * (len(title) + 8) + "\n")
    
    def check_token_details(
        self, 
        token_details: Optional[List[Dict[str, Any]]], 
        type_name: str = "Tokens",
        verbose: bool = True
    ) -> bool:
        """
        Checks token details and returns whether all are greedy.
        
        Args:
            token_details: List of token detail dictionaries
            type_name: Name for the token type (e.g., "Prompt", "Completion")
            verbose: Whether to print detailed information
            
        Returns:
            True if all tokens with rank info are greedy (rank 1), False otherwise
        """
        if not token_details:
            if verbose:
                print(f"  ({type_name} details not available or empty)")
            return True
        
        all_greedy = True
        if verbose:
            print(f"  {type_name} Details:")
            
        for td in token_details:
            token_id = td.get('token_id', 'N/A')
            text = td.get('text', '')
            logprob = td.get('logprob', 0.0)
            rank = td.get('rank')
            alternatives = td.get('alternatives', [])
            
            is_greedy_char = '✔' if rank == 1 else ('✘' if rank is not None else '?')
            
            if verbose:
                print(f"    ID: {str(token_id):<5} Text: '{text}' "
                      f"Logprob: {logprob:<8.4f} Rank: {str(rank):<3} "
                      f"Greedy: {is_greedy_char}")
                
                # Show alternatives if available
                if alternatives:
                    print(f"      Alternatives:")
                    for alt in alternatives[:3]:  # Show top 3 alternatives
                        alt_text = alt.get('text', '')
                        alt_logprob = alt.get('logprob', 0.0)
                        alt_rank = alt.get('rank', 'N/A')
                        print(f"        - '{alt_text}' (logprob: {alt_logprob:.4f}, rank: {alt_rank})")
            
            if rank is not None and rank != 1:
                all_greedy = False
                
        return all_greedy
    
    def make_request(
        self, 
        endpoint: str, 
        payload: Dict[str, Any],
        method: str = "POST"
    ) -> Union[Dict[str, Any], None]:
        """
        Makes an HTTP request to the specified endpoint.
        
        Args:
            endpoint: API endpoint path
            payload: Request payload
            method: HTTP method (default: POST)
            
        Returns:
            Response data as dictionary or None on error
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == "POST":
                response = requests.post(url, json=payload)
            else:
                raise ValueError(f"Unsupported method: {method}")
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response content: {e.response.text}")
            return None
            
    def verify_completion(
        self, 
        model: str,
        prompt: str,
        max_tokens: int = 10,
        prompt_logprobs: int = 5,
        verbose: bool = True
    ) -> bool:
        """
        Tests verified completion endpoint.
        
        Returns:
            True if test passed, False otherwise
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "prompt_logprobs": prompt_logprobs,
            "logprobs": 5
        }
        
        if verbose:
            print("=== Testing /v1/completions/verified ===")
            self.print_json(payload, "Request Payload")
            
        data = self.make_request("/completions/verified", payload)
        if data is None:
            return False
            
        if verbose:
            self.print_json(data, "Verified Completion Response")
            
        if not data.get("choices"):
            print("ERROR: No choices in response.")
            return False
            
        choice = data["choices"][0]
        
        if verbose:
            print(f"\nGenerated Text: {choice.get('text')}")
            print("\nPrompt Token Details:")
            self.check_token_details(choice.get("prompt_token_details"), "Prompt")
            print("\nCompletion Token Details:")
            
        all_greedy = self.check_token_details(
            choice.get("completion_token_details"), 
            "Completion",
            verbose=verbose
        )
        
        if verbose:
            if all_greedy:
                print("SUCCESS: All completion tokens are greedy (rank 1).")
            else:
                print("WARNING: Not all completion tokens were rank 1.")
                
        return all_greedy
    
    def verify_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 15,
        prompt_logprobs: int = 3,
        verbose: bool = True
    ) -> bool:
        """
        Tests verified chat completion endpoint.
        
        Returns:
            True if test passed, False otherwise
        """
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "prompt_logprobs": prompt_logprobs,
            "logprobs": True,
            "top_logprobs": 3
        }
        
        if verbose:
            print("=== Testing /v1/chat/completions/verified ===")
            self.print_json(payload, "Request Payload")
            
        data = self.make_request("/chat/completions/verified", payload)
        if data is None:
            return False
            
        if verbose:
            self.print_json(data, "Verified Chat Completion Response")
            
        if not data.get("choices"):
            print("ERROR: No choices in response.")
            return False
            
        choice = data["choices"][0]
        message = choice.get("message", {})
        
        if verbose:
            print(f"\nAssistant's Message: {message.get('content')}")
            print("\nPrompt Token Details:")
            self.check_token_details(choice.get("prompt_token_details"), "Prompt")
            print("\nCompletion Token Details:")
            
        all_greedy = self.check_token_details(
            choice.get("completion_token_details"),
            "Completion",
            verbose=verbose
        )
        
        if verbose:
            if all_greedy:
                print("SUCCESS: All completion tokens are greedy (rank 1).")
            else:
                print("WARNING: Not all completion tokens were rank 1.")
                
        return all_greedy
    
    def verify_decoding(
        self,
        model: str,
        prompt: Union[str, List[int]],
        completion: Union[str, List[int]],
        check_greedy: bool = True,
        greedy_logprob_threshold: float = 0.001,
        verbose: bool = True
    ) -> Optional[bool]:
        """
        Tests verify_decoding endpoint.
        
        Returns:
            True if verified as greedy, False if not, None on error
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "completion": completion,
            "check_greedy": check_greedy,
            "greedy_logprob_threshold": greedy_logprob_threshold
        }
        
        if verbose:
            print("=== Testing /v1/verify_decoding ===")
            self.print_json(payload, "Request Payload")
            
        data = self.make_request("/verify_decoding", payload)
        if data is None:
            return None
            
        if verbose:
            self.print_json(data, "Verify Decoding Response")
            print(f"\nOverall Greedy Verification: {data.get('is_verified_greedy')}")
            
            if data.get("verification_details"):
                print("\nVerification Details:")
                for detail in data["verification_details"]:
                    token_id = detail.get('token_id')
                    text = detail.get('text', '')
                    is_greedy = detail.get('is_greedy_choice')
                    error = detail.get('error_message')
                    
                    status = '✔' if is_greedy else ('✘' if is_greedy is False else '?')
                    print(f"  Token {token_id} ('{text}'): {status}")
                    if error:
                        print(f"    Error: {error}")
                        
        return data.get('is_verified_greedy')
    
    def batch_verify_decoding(
        self,
        model: str,
        verification_pairs: List[Dict[str, Union[str, List[int]]]],
        check_greedy: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Tests batch verify_decoding endpoint.
        
        Args:
            model: Model name
            verification_pairs: List of dicts with 'prompt' and 'completion' keys
            check_greedy: Whether to check for greedy decoding
            verbose: Whether to print detailed output
            
        Returns:
            Summary statistics dictionary
        """
        requests = [
            {
                "model": model,
                "prompt": pair["prompt"],
                "completion": pair["completion"],
                "check_greedy": check_greedy
            }
            for pair in verification_pairs
        ]
        
        payload = {"requests": requests}
        
        if verbose:
            print("=== Testing /v1/verify_decoding/batch ===")
            print(f"Verifying {len(requests)} prompt-completion pairs...")
            
        data = self.make_request("/verify_decoding/batch", payload)
        if data is None:
            return {"error": "Request failed"}
            
        if verbose:
            summary = data.get("summary", {})
            print(f"\nBatch Verification Summary:")
            print(f"  Total: {summary.get('total', 0)}")
            print(f"  Verified: {summary.get('verified', 0)}")
            print(f"  Not Verified: {summary.get('not_verified', 0)}")
            print(f"  Failed: {summary.get('failed', 0)}")
            
            # Show details for non-verified or failed items
            results = data.get("results", [])
            for i, result in enumerate(results):
                if isinstance(result, dict):
                    if result.get("object") == "text.verification":
                        if not result.get("is_verified_greedy"):
                            print(f"\n  Pair {i} not verified as greedy")
                    else:  # Error response
                        print(f"\n  Pair {i} failed: {result.get('message', 'Unknown error')}")
                        
        return data.get("summary", {})


# Convenience function for quick testing
def run_basic_verification_tests(
    host: str = "localhost",
    port: int = 8000,
    model: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
):
    """Runs basic verification tests against the vLLM server."""
    client = VerificationClient(host, port)
    
    print(f"Testing verification endpoints on {host}:{port}")
    print(f"Using model: {model}\n")
    
    # Test 1: Verified Completion
    client.verify_completion(
        model=model,
        prompt="The capital of France is",
        max_tokens=5
    )
    
    print("\n" + "="*60 + "\n")
    
    # Test 2: Verified Chat Completion
    client.verify_chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ],
        max_tokens=10
    )
    
    print("\n" + "="*60 + "\n")
    
    # Test 3: Verify Decoding
    client.verify_decoding(
        model=model,
        prompt="The sky is",
        completion=" blue",
        check_greedy=True
    )
    
    print("\n" + "="*60 + "\n")
    
    # Test 4: Batch Verify Decoding
    client.batch_verify_decoding(
        model=model,
        verification_pairs=[
            {"prompt": "1 + 1 =", "completion": " 2"},
            {"prompt": "The sun is", "completion": " hot"},
            {"prompt": "Water is", "completion": " wet"}
        ]
    )


if __name__ == "__main__":
    # Run tests if script is executed directly
    run_basic_verification_tests()
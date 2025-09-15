#!/usr/bin/env python3
"""
Unified runner for vLLM extended verified endpoints.

Scenarios:
- completions-verified
- chat-verified
- verify-decoding
- long-chat
- all

Outputs:
- --format pretty|json
- Non-zero exit on any failure (CI-friendly)
"""

import argparse
import json
import sys
from typing import Any, Dict, List, Optional
import requests

DEFAULT_SERVER_URL = "http://localhost:8000/v1"
DEFAULT_MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEFAULT_PROMPT = "Hello, world! My name is"
SESSION = requests.Session()

def _print(section: str, data: Any, out_format: str, quiet: bool, status: Optional[str] = None, meta: Optional[Dict[str, Any]] = None):
    if out_format == "json":
        obj = {"section": section, "status": status, "data": data}
        if meta:
            obj["meta"] = meta
        print(json.dumps(obj))
    elif not quiet:
        header = f"--- {section} ---"
        print(header)
        if isinstance(data, (dict, list)):
            print(json.dumps(data, indent=2))
        else:
            print(str(data))
        print("-" * len(header))

def request_json(url: str, payload: Optional[dict] = None) -> Dict[str, Any]:
    r = SESSION.post(url, json=payload, headers={"Content-Type": "application/json"})
    r.raise_for_status()
    return r.json()

def ensure(condition: bool, msg: str, errors: List[str]):
    if not condition:
        errors.append(msg)

def has_rank1(details: Optional[List[dict]]) -> bool:
    if not details:
        return True
    ok = True
    for td in details:
        rank = td.get("rank")
        if rank is not None and rank != 1:
            ok = False
            break
    return ok

def check_chat_verified(server_url: str, model: str, prompt: str, max_tokens: int, out_format: str, quiet: bool, top_logprobs: int = 1) -> List[str]:
    errors: List[str] = []
    endpoint = f"{server_url}/chat/completions/verified"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "prompt_logprobs": top_logprobs,
        "logprobs": True,
        "top_logprobs": top_logprobs
    }
    data = request_json(endpoint, payload)
    _print("chat/completions/verified response", data, out_format, quiet)

    ensure("choices" in data and data["choices"], "missing or empty choices", errors)
    if not errors:
        ch0 = data["choices"][0]
        # Flexible support: fields may live per-choice
        ptd = ch0.get("prompt_token_details") or data.get("prompt_token_details")
        ctd = ch0.get("completion_token_details") or data.get("completion_token_details")
        ensure(ch0.get("message", {}).get("content") is not None, "missing message.content", errors)
        ensure(ptd is not None, "missing prompt_token_details", errors)
        ensure(ctd is not None, "missing completion_token_details", errors)
        ensure(has_rank1(ctd), "completion tokens not greedy (rank != 1) with temperature=0", errors)
    return errors

def check_completions_verified(server_url: str, model: str, prompt: str, max_tokens: int, out_format: str, quiet: bool, top_logprobs: int = 1) -> List[str]:
    errors: List[str] = []
    endpoint = f"{server_url}/completions/verified"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "prompt_logprobs": top_logprobs,
        "logprobs": top_logprobs
    }
    data = request_json(endpoint, payload)
    _print("completions/verified response", data, out_format, quiet)

    ensure("choices" in data and data["choices"], "missing or empty choices", errors)
    if not errors:
        ch0 = data["choices"][0]
        ensure(ch0.get("text") is not None, "missing text", errors)
        ptd = ch0.get("prompt_token_details") or data.get("prompt_token_details")
        ctd = ch0.get("completion_token_details") or data.get("completion_token_details")
        ensure(ptd is not None, "missing prompt_token_details", errors)
        ensure(ctd is not None, "missing completion_token_details", errors)
        ensure(has_rank1(ctd), "completion tokens not greedy (rank != 1) with temperature=0", errors)
    return errors

def tokenize(server_url: str, model: str, text: str, add_special_tokens: bool = False) -> List[int]:
    # Note: tokenize is typically outside /v1, keep compatibility with docs
    url = server_url.rstrip("/v1") + "/tokenize"
    data = request_json(url, {"model": model, "prompt": text, "add_special_tokens": add_special_tokens})
    return data.get("tokens") or data.get("token_ids") or []

def check_verify_decoding(server_url: str, model: str, base_prompt: str, sample_completion_text: Optional[str],
                          out_format: str, quiet: bool) -> List[str]:
    errors: List[str] = []
    endpoint = f"{server_url}/verify_decoding"

    # If completion text not provided, get one from text completions (NOT chat)
    # Using the same decoding context (plain text) avoids chat template/token
    # mismatches when verifying greedy decoding.
    if sample_completion_text is None:
        aux = request_json(f"{server_url}/completions/verified", {
            "model": model,
            "prompt": base_prompt,
            "max_tokens": 50,
            "temperature": 0.0,
            "prompt_logprobs": 1,
            "logprobs": 1
        })
        choices = aux.get("choices", [])
        if not choices:
            errors.append("verify_decoding prep: could not get greedy completion from /completions/verified")
            return errors
        sample_completion_text = choices[0].get("text", "")

    # Tokenize for token-ID path
    prompt_ids = tokenize(server_url, model, base_prompt, add_special_tokens=False)
    comp_ids = tokenize(server_url, model, sample_completion_text, add_special_tokens=False)

    # Case 1: token ids path (expect True)
    resp1 = request_json(endpoint, {
        "model": model,
        "prompt": prompt_ids,
        "completion": comp_ids,
        "check_greedy": True
    })
    _print("verify_decoding case1 (ids)", resp1, out_format, quiet)
    ensure(resp1.get("is_verified_greedy") is True, "verify_decoding ids path not verified", errors)

    # Case 2: text path (expect True)
    resp2 = request_json(endpoint, {
        "model": model,
        "prompt": base_prompt,
        "completion": sample_completion_text,
        "check_greedy": True
    })
    _print("verify_decoding case2 (text)", resp2, out_format, quiet)
    ensure(resp2.get("is_verified_greedy") is True, "verify_decoding text path not verified", errors)

    # Case 3: empty completion (expect True)
    resp3 = request_json(endpoint, {"model": model, "prompt": base_prompt, "completion": ""})
    _print("verify_decoding case3 (empty text)", resp3, out_format, quiet)
    ensure(resp3.get("is_verified_greedy") is True, "empty completion should verify", errors)

    # Case 4: deliberate non-greedy (expect False)
    disruptive = " ZYXWVU "
    resp4 = request_json(endpoint, {
        "model": model,
        "prompt": base_prompt,
        "completion": disruptive + sample_completion_text,
        "check_greedy": True
    })
    _print("verify_decoding case4 (non-greedy)", resp4, out_format, quiet)
    ensure(resp4.get("is_verified_greedy") is False, "non-greedy divergence not detected", errors)

    return errors

def check_long_chat(server_url: str, model: str, out_format: str, quiet: bool) -> List[str]:
    errors: List[str] = []
    story_prompt = "Write a short story about a curious robot exploring a mysterious ancient library. The story should be at least 100 words long."
    # Get greedy chat with token details
    chat = request_json(f"{server_url}/chat/completions/verified", {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a creative and skilled storyteller."},
            {"role": "user", "content": story_prompt}
        ],
        "max_tokens": 250,
        "temperature": 0.0,
        "prompt_logprobs": 1,
        "logprobs": True,
        "top_logprobs": 1
    })
    _print("long-chat response", chat, out_format, quiet)

    choices = chat.get("choices", [])
    ensure(bool(choices), "long-chat: missing choices", errors)
    if errors:
        return errors

    ch0 = choices[0]
    story = ch0.get("message", {}).get("content")
    ensure(bool(story), "long-chat: empty story", errors)

    ptd = ch0.get("prompt_token_details") or chat.get("prompt_token_details")
    ensure(bool(ptd), "long-chat: missing prompt_token_details", errors)
    if ptd:
        p_ids = [d["token_id"] for d in ptd if "token_id" in d]
        verify = request_json(f"{server_url}/verify_decoding", {
            "model": model,
            "prompt": p_ids,
            "completion": story,
            "check_greedy": True
        })
        _print("long-chat verify_decoding", verify, out_format, quiet)
        ensure(verify.get("is_verified_greedy") is True, "long-chat story not verified as greedy", errors)
    return errors

def main():
    ap = argparse.ArgumentParser(description="Run verified endpoint checks.")
    ap.add_argument("--server-url", type=str, default=DEFAULT_SERVER_URL)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    ap.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    ap.add_argument("--max-tokens", type=int, default=50)
    ap.add_argument("--format", choices=["pretty", "json"], default="pretty")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--scenario", choices=["completions-verified","chat-verified","verify-decoding","long-chat","all"], default="all")
    args = ap.parse_args()

    failures: List[str] = []

    if args.scenario in ("completions-verified", "all"):
        errs = check_completions_verified(args.server_url, args.model, args.prompt, args.max_tokens, args.format, args.quiet)
        if errs: _print("completions-verified FAIL", errs, args.format, False, status="FAIL")
        failures += [f"[completions-verified] {e}" for e in errs]
        if not errs: _print("completions-verified", "PASS", args.format, False, status="PASS")

    if args.scenario in ("chat-verified", "all"):
        errs = check_chat_verified(args.server_url, args.model, args.prompt, args.max_tokens, args.format, args.quiet)
        if errs: _print("chat-verified FAIL", errs, args.format, False, status="FAIL")
        failures += [f"[chat-verified] {e}" for e in errs]
        if not errs: _print("chat-verified", "PASS", args.format, False, status="PASS")

    if args.scenario in ("verify-decoding", "all"):
        errs = check_verify_decoding(args.server_url, args.model, args.prompt, None, args.format, args.quiet)
        if errs: _print("verify-decoding FAIL", errs, args.format, False, status="FAIL")
        failures += [f"[verify-decoding] {e}" for e in errs]
        if not errs: _print("verify-decoding", "PASS", args.format, False, status="PASS")

    if args.scenario in ("long-chat", "all"):
        errs = check_long_chat(args.server_url, args.model, args.format, args.quiet)
        if errs: _print("long-chat FAIL", errs, args.format, False, status="FAIL")
        failures += [f"[long-chat] {e}" for e in errs]
        if not errs: _print("long-chat", "PASS", args.format, False, status="PASS")

    if failures:
        if args.format == "json":
            print(json.dumps({"summary": {"passed": False, "failures": failures}}))
        else:
            print("\nFAILURES:")
            for f in failures:
                print(f" - {f}")
        sys.exit(1)
    else:
        if args.format == "json":
            print(json.dumps({"summary": {"passed": True}}))
        else:
            print("\nAll tests PASSED.")
        sys.exit(0)

if __name__ == "__main__":
    main()

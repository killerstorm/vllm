# Extended OpenAI-Compatible API Endpoints for vLLM

This document describes additional API endpoints provided by vLLM that extend the standard OpenAI API, primarily focused on token-level information and verification capabilities.

## 1. Chat Completions with Token IDs

### Endpoint: `POST /v1/chat/completions/verified`

This endpoint behaves similarly to the standard `/v1/chat/completions` endpoint but is tailored for scenarios requiring direct access to token IDs and ensuring greedy generation for verification purposes.

**Purpose:**
- Generate chat completions with temperature forced to 0.0 (greedy decoding).
- Include token IDs for both the prompt and the generated completion(s) in the response.
- Ensure log probabilities for prompt and completion tokens are returned.

**Request Body:**

The request body largely follows the OpenAI Chat Completions API.

```json
{
  \"model\": \"string (model name)\",
  \"messages\": [
    {\"role\": \"user\", \"content\": \"Hello, what is the capital of France?\"}
    // ... other messages
  ],
  \"temperature\": \"float (e.g., 0.7, will be overridden to 0.0 by the server)\",
  \"max_tokens\": \"integer (e.g., 100)\",
  // Other standard chat completion parameters like top_p, stop, presence_penalty, etc.,
  // might be supported, but 'temperature' is always overridden.
  // The server will implicitly enable logprobs and prompt_logprobs.
}
```

**Server Behavior:**
- **Greedy Generation:** The server will override any provided `temperature` to `0.0` and use `top_k=1` (or equivalent) to ensure deterministic, greedy output.
- **Logprobs Enforcement:** The server will behave as if `logprobs: true` and `prompt_logprobs: <top_n_value>` (e.g., 5) were implicitly part of the request, ensuring these fields are populated in the response. `return_tokens_as_ids` is also implicitly enabled.

**Response Body:**

The response body extends the standard OpenAI Chat Completions response.

```json
{
  \"id\": \"string (e.g., chatcmpl-xxxxxxxx)\",
  \"object\": \"string (e.g., chat.completion)\",
  \"created\": \"integer (Unix timestamp)\",
  \"model\": \"string (model name used)\",
  \"choices\": [
    {
      \"index\": \"integer\",
      \"message\": {
        \"role\": \"assistant\",
        \"content\": \"string (generated text, e.g., The capital of France is Paris.)\"
      },
      \"logprobs\": { // Logprobs for the completion tokens
        \"content\": [
          {
            \"token\": \"string (decoded token text)\",
            \"logprob\": \"float\",
            \"bytes\": \"Optional[list[integer]]\",
            \"top_logprobs\": [ // Logprobs of top N tokens at this position
              {\"token\": \"string\", \"logprob\": \"float\", \"bytes\": \"Optional[list[integer]]\"},
              // ... more tokens
            ]
          },
          // ... more completion tokens
        ]
      },
      \"finish_reason\": \"string (e.g., stop, length)\"
    }
    // ... other choices if n > 1 (though greedy implies one unique choice)
  ],
  \"usage\": {
    \"prompt_tokens\": \"integer\",
    \"completion_tokens\": \"integer\",
    \"total_tokens\": \"integer\"
  },
  // Extended fields:
  \"prompt_token_ids\": \"Optional[list[integer]] (e.g., [123, 456, 789])\",
  \"completion_token_ids\": \"Optional[list[list[integer]]] (e.g., [[101, 102]])\",
  \"prompt_logprobs\": \"Optional[list[Optional[dict[integer, LogprobObject]]]]\" 
                       // Logprobs for each prompt token.
                       // Each dict maps a token_id to its LogprobObject:
                       //   { \"logprob\": float, \"token_str\": string, \"bytes\": Optional[list[int]] }
                       // This represents the distribution for predicting the *next* token in the prompt,
                       // or the first completion token after the last prompt token.
                       // The outer list corresponds to each position in the prompt.
}
```
**Key Extended Fields in Response:**
- `prompt_token_ids`: A list of integers representing the token IDs of the input prompt.
- `completion_token_ids`: A list of lists of integers. Each inner list contains the token IDs for one generated completion choice.
- `prompt_logprobs`: Provides log probabilities for tokens in the prompt. `prompt_logprobs[i]` contains a dictionary mapping potential next token IDs (integer) to their log probabilities (and decoded string representations) given the prompt tokens up to `prompt_token_ids[i]`. The server determines how many top logprobs to return for each position (e.g., top 5).

## 2. Greedy Completion Verification

### Endpoint: `POST /v1/verify_decoding`

This endpoint allows for verifying whether a given completion (text or token IDs) for a given prompt (text or token IDs) is a result of greedy token generation by the specified model.

**Purpose:**
- Determine if a candidate completion follows the model's highest-probability path.
- Provide detailed log C probability information to understand any discrepancies.

**Request Body (`GreedyVerificationRequest`):**

```json
{
  \"model\": \"Optional[string (model name)]\",
  \"prompt\": \"Optional[string (prompt text)]\",
  \"completion\": \"Optional[string (completion text to verify)]\",
  \"prompt_token_ids\": \"Optional[list[integer]]\",
  \"completion_token_ids\": \"Optional[list[integer]]\",
  \"add_special_tokens\": \"boolean (default: true, applies if 'prompt' text is given)\",
  \"request_id\": \"Optional[string (defaults to a random UUID)]\"
}
```
**Important:**
- The client must provide *either* the text pair (`prompt`, `completion`) *or* the token ID pair (`prompt_token_ids`, `completion_token_ids`).
- `add_special_tokens`: If `prompt` (text) is provided, this flag controls whether special tokens (e.g., BOS) are added during its tokenization. This is generally not applied to the `completion` text during its tokenization.

**Server Behavior:**
1.  **Tokenization (if text provided):** If text inputs are given, the server tokenizes `prompt` (respecting `add_special_tokens`) and `completion` (typically without adding special tokens).
2.  **Empty Completion:** An empty completion is considered trivially verified (`verified: true`).
3.  **Length Check:** The server checks if the combined length of prompt and completion tokens exceeds the model's maximum sequence length. If so, an error is returned.
4.  **Inference with Logprobs:** The server performs an internal inference run. It effectively feeds the sequence `prompt_token_ids + completion_token_ids` to the model. For each token in this combined sequence (starting from the first token of the prompt), it calculates the log probabilities of the *next* token. Crucially, for this internal run, sampling is set to be greedy (e.g., temperature 0, top_k 1) and requests logprobs for a number of top alternative tokens at each step (e.g., top 5).
5.  **Verification:**
    - The server iterates through the provided `completion_token_ids`.
    - For each actual token `completion_token_ids[i]` at position `i` in the completion, it compares it against the token that had the highest log probability (the greedy choice) based on the internal inference run (i.e., the model's prediction after processing `prompt_token_ids + completion_token_ids[0...i-1]`).
    - If any token in the provided `completion_token_ids` does not match the model's greedy prediction for that step, `verified` is set to `false`.

**Response Body (`GreedyVerificationResponse`):**

```json
{
  \"id\": \"string (request ID)\",
  \"object\": \"string (fixed: 'text_verification')\",
  \"created\": \"integer (Unix timestamp)\",
  \"model\": \"string (model name used)\",
  \"verified\": \"boolean (true if completion is greedy, false otherwise)\",
  \"usage\": {
    \"prompt_tokens\": \"integer\",
    \"completion_tokens\": \"integer\",
    \"total_tokens\": \"integer\"
  },
  \"logprobs\": \"Optional[VerificationLogprobsObject]\"
}
```

**`VerificationLogprobsObject` Structure:**
This object provides detailed data used for and resulting from the verification process.

```json
{
  \"tokens\": \"list[string] (actual completion tokens provided, decoded)\",
  \"token_ids\": \"list[integer] (actual completion token IDs provided)\",
  \"top_logprobs\": \"list[dict[string, float]]\", 
                  // For each actual completion token at index `i`, this is a dict 
                  // of {token_string: logprob} for the top N tokens predicted by 
                  // the model after processing prompt + completion[0...i-1].
                  // Includes the actual token[i]'s logprob.
  \"expected_tokens\": \"list[string] (greedy tokens predicted by the model, decoded)\",
  \"expected_token_ids\": \"list[integer] (greedy token IDs predicted by model)\",
  \"failure_idx\": \"Optional[integer] (0-indexed, if not verified, the first index of mismatch in completion)\",
  
  // Full sequence information for context/debugging:
  \"prompt_tokens\": \"Optional[list[string]] (decoded prompt tokens)\",
  \"prompt_token_ids\": \"Optional[list[integer]] (prompt token IDs)\",
  \"full_sequence_tokens\": \"Optional[list[str]] (decoded prompt + actual completion tokens)\",
  \"full_sequence_token_ids\": \"Optional[list[int]] (prompt + actual completion token IDs)\",
  \"full_sequence_logprobs\": \"Optional[list[dict[string, float]]]\"
                            // Logprobs for each step in the full_sequence. 
                            // full_sequence_logprobs[j] contains {token_string: logprob} 
                            // for predicting full_sequence_tokens[j+1] given tokens up to j.
}
```

**Key fields in `VerificationLogprobsObject`:**
- `tokens`, `token_ids`: The completion sequence that was submitted for verification.
- `top_logprobs`: At each step `i` of the provided completion, what were the model's top predicted next tokens (and their logprobs), after having processed the prompt and the first `i-1` tokens of the provided completion.
- `expected_tokens`, `expected_token_ids`: The sequence the model *would have* generated greedily starting from the prompt.
- `failure_idx`: If `verified` is `false`, this points to the first token in `tokens`/`token_ids` that diverged from `expected_tokens`/`expected_token_ids`.
- `full_sequence_tokens`, `full_sequence_token_ids`, `full_sequence_logprobs`: These provide a complete trace of the tokens and associated next-token-prediction logprobs for the *entire input sequence* (prompt + completion) that was fed to the model during the verification inference step. This is useful for detailed analysis of why a sequence was or wasn't considered greedy. 
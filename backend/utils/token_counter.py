# backend/utils/token_counter.py

import tiktoken


class TokenCounter:
    """
    Utility to calculate token usage for LLM calls.
    Works with OpenAI-compatible tokenizers.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        # Fallback-safe encoding
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a single string.
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def count_prompt_and_completion(
        self,
        prompt: str,
        completion: str,
    ) -> dict:
        """
        Count tokens separately for prompt and completion.
        """
        prompt_tokens = self.count_tokens(prompt)
        completion_tokens = self.count_tokens(completion)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
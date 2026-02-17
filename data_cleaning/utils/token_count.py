"""Count input tokens for cost reporting (system + user prompt)."""


def count_tokens(text: str) -> int:
    """Approximate token count. Uses tiktoken if available, else ~4 chars per token."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)

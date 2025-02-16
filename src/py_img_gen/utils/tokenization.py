from transformers import BatchEncoding, CLIPTokenizer


def tokenize_prompt(prompt: str, tokenizer: CLIPTokenizer) -> BatchEncoding:
    """Tokenize the prompt using the given tokenizer.

    Args:
        prompt (str): The prompt to tokenize.
        tokenizer (CLIPTokenizer): The tokenizer to use.

    Returns:
        BatchEncoding: The tokenized prompt.
    """

    return tokenizer(
        text=prompt,
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )

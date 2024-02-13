import math
import torch
from pathlib import Path
from torch import nn
from torch import Tensor
from transformers import AutoTokenizer, GPT2Tokenizer


class PositionalEncoding(nn.Module):
    """From https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor, account_for_lang_token: bool = False) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
            account_for_lang_token: bool, optional (default: False)
        """
        if account_for_lang_token:
            # Shift the positional encoding by 1
            x = x + self.pe[1 : x.size(0) + 1]  # noqa
        else:
            x = x + self.pe[: x.size(0)]
        return self.dropout(x)


def filter_by_token_length(
    code_tokenizer,
    english_tokenizer,
    example,
    max_code_length=512,
    max_docstring_length=512,
):
    """Filter out examples that are too long for the model to handle."""
    tokenized_code = code_tokenizer(example["code"], return_length=True)
    tokenized_docstring = english_tokenizer(example["docstring"], return_length=True)

    def less_than_max_len_list(x, ml):
        return x["length"][0] <= ml

    def less_than_max_len(x, ml):
        return x["length"] <= ml

    return less_than_max_len(tokenized_code, max_code_length) and less_than_max_len(
        tokenized_docstring, max_docstring_length
    )


def generate_square_subsequent_mask(sz: int, device="cpu") -> Tensor:
    """
    Generate a square mask for the sequence, masking out subsequent positions.

    The mask is an upper triangular matrix with zeros on the diagonal and above
    and `-inf` below the diagonal. This mask is used in the decoder to prevent
    the model from attending to future tokens in the sequence when predicting
    the current token.

    Args:
        sz (int): The size of the square mask.

    Returns:
        Tensor: The mask as a 2D tensor of shape (sz, sz), where the diagonal and
                elements above the diagonal are 0, and the elements below the
                diagonal are -inf.

    probably flip this
    """
    return nn.Transformer().generate_square_subsequent_mask(sz, device=device)


def create_padding_mask(src: Tensor, pad_idx: int) -> Tensor:
    """
    Create padding mask for the given input tensor.

    Args:
        src (Tensor): The input tensor (shape: [batch_size, seq_length]).
        pad_idx (int): The padding token index.

    Returns:
        Tensor: The padding mask (shape: [batch_size, seq_length]).
    """
    # Create a mask where True corresponds to the padding tokens
    mask = (src == pad_idx).transpose(0, 1)
    return mask.to(torch.float32)


def add_pad_token(tokenizer):
    pad_token = "<PAD>"
    tokenizer.add_tokens([pad_token], special_tokens=True)
    tokenizer.pad_token = pad_token
    return tokenizer


def get_english_tokenizer():
    """
    Get the english language tokenizer for the model.
    From https://huggingface.co/gpt2
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer = add_pad_token(tokenizer)
    return tokenizer


def add_language_tokens(tokenizer):
    lang_tokens = [
        "<PYTHON>",
        "<JAVA>",
        "<GO>",
        "<JAVASCRIPT>",
        "<RUBY>",
        "<PHP>",
    ]
    tokenizer.add_tokens(lang_tokens, special_tokens=True)

    # Create a dictionary to map language names to special tokens
    lang_to_token = {
        "python": "<PYTHON>",
        "java": "<JAVA>",
        "go": "<GO>",
        "javascript": "<JAVASCRIPT>",
        "ruby": "<RUBY>",
        "php": "<PHP>",
    }
    tokenizer.lang_to_token = lang_to_token
    return tokenizer


def get_code_tokenizer():
    """
    Get the tokenizer for the model.
    From https://github.com/salesforce/CodeGen
    """
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-2B-mono")
    tokenizer = add_pad_token(tokenizer)
    tokenizer = add_language_tokens(tokenizer)

    return tokenizer


def save_model(model, path):
    """Save pytorch model to path"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path, device):
    """Load pytorch model from path"""
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

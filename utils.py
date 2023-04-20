import math
import torch
from pathlib import Path
from torch import nn
from torch import Tensor
from transformers import AutoTokenizer


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


def filter_by_token_length(tokenizer, example, max_length=512):
    """Filter out examples that are too long for the model to handle."""
    tokenized_code = tokenizer(example["code"], return_length=True)
    tokenized_docstring = tokenizer(example["docstring"], return_length=True)

    def less_than_max_len(x):
        return x["length"][0] <= max_length

    return less_than_max_len(tokenized_code) and less_than_max_len(tokenized_docstring)


def generate_square_subsequent_mask(sz: int) -> Tensor:
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
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def get_tokenizer():
    """
    Get the tokenizer for the model.
    From https://github.com/salesforce/CodeGen
    """
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-2B-mono")
    tokenizer.pad_token = tokenizer.eos_token
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


def save_model(model, path):
    """Save pytorch model to path"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(cls, path, *args, **kwargs):
    """Load pytorch model from path"""
    model = cls(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

import math
import torch
from typing import Optional
from torch.nn import (
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
)
from torch import nn
from torch import Tensor
from utils import PositionalEncoding


class TransformerEncoderModel(nn.Module):
    """
    Transformer Encoder Model.

    This class implements the encoder part of the transformer architecture.
    """

    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model

        self.init_weights()

    def init_weights(self) -> None:
        """There might be a more efficient way to do this, e.g. use
        ``torch.nn.init.xavier_uniform_``."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_padding_mask: Tensor) -> Tensor:
        """
        Forward pass of the transformer encoder model.

        Args:
            src (Tensor): The input tensor.
            src_padding_mask (Tensor): The source padding mask tensor. This is
            to prevent the model from attending to the padding token.

        Returns:
            Tensor: The output tensor.
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=src_padding_mask)
        return output


class TransformerDecoderModel(nn.Module):
    """
    Transformer Decoder Model.

    This class implements the decoder part of the transformer architecture.
    """

    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        decoder_layers = TransformerDecoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor,
        lang_token: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of the transformer decoder model.

        Args:
            tgt (Tensor): The input tensor.
            memory (Tensor): The memory tensor from the encoder.
            tgt_mask (Tensor): The target mask tensor.
            lang_token (Optional[Tensor]): The optional language token tensor.

        Returns:
            Tensor: The output tensor.
        """
        tgt_embed = self.embedding(tgt) * math.sqrt(
            self.d_model
        )  # (seq_len, batch_size, d_model)

        if lang_token is not None:
            lang_token_embed = (
                self.embedding(lang_token) * math.sqrt(self.d_model)
            ).permute(
                1, 0, 2
            )  # (1, batch_size, d_model)

            # Prepend the value_to_prepend tensor to the original tensor
            tgt_embed = torch.cat(
                (lang_token_embed, tgt_embed), dim=0
            )  # (seq_len+2, batch_size, d_model)
            tgt_embed = tgt_embed[:-1, :, :]  # (seq_len+1, batch_size, d_model)
            tgt_embed = self.pos_encoder(
                tgt_embed, account_for_lang_token=True
            )  # (seq_len+1, batch_size, d_model)
        else:
            tgt_embed = self.pos_encoder(tgt_embed)  # (seq_len, batch_size, d_model)
        output = self.transformer_decoder(
            tgt_embed, memory, tgt_mask=tgt_mask
        )  # (seq_len, batch_size, d_model) or (seq_len+1, batch_size, d_model)
        output = self.linear(
            output
        )  # (seq_len, batch_size, vocab_size) or (seq_len+1, batch_size, vocab_size)
        return output

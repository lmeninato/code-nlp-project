import math
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
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model

        self.init_weights()

    def init_weights(self) -> None:
        """There might be a more efficient way to do this, e.g. use
        ``torch.nn.init.xavier_uniform_``."""
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Forward pass of the transformer encoder model.

        Args:
            src (Tensor): The input tensor.
            src_mask (Tensor): The source mask tensor.

        Returns:
            Tensor: The output tensor.
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
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
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

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
        tgt_embed = self.encoder(tgt) * math.sqrt(self.d_model)

        if lang_token is not None:
            lang_token_embed = self.encoder(lang_token) * math.sqrt(self.d_model)
            tgt_embed[:, -1, :] = lang_token_embed.squeeze(1)
            tgt_embed = self.pos_encoder(
                tgt_embed, account_for_lang_token=True
            )  # 16 x 513 x 256
        else:
            tgt_embed = self.pos_encoder(tgt_embed)  # 16 x 513 x 256

        output = self.transformer_decoder(tgt_embed, memory, tgt_mask)
        output = self.decoder(output)
        return output

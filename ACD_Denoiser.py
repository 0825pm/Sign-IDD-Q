# coding: utf-8
import torch.nn as nn
import torch
import math
from torch import Tensor
import einops

from helpers import freeze_params, subsequent_mask
from transformer_layers import PositionalEncoding, TransformerDecoderLayer

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ACD_Denoiser(nn.Module):

    def __init__(self,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 hidden_size: int = 512,
                 ff_size: int = 2048,
                 dropout: float = 0.1,
                 emb_dropout: float = 0.1,
                 vocab_size: int = 1,
                 freeze: bool = False,
                 trg_size: int = 150,
                 decoder_trg_trg_: bool = True,
                 **kwargs):
        super(ACD_Denoiser, self).__init__()

        # self.in_feature_size = trg_size + (trg_size // 3) * 4
        self.in_feature_size = trg_size
        self.out_feature_size = trg_size

        self.pos_drop = nn.Dropout(p=emb_dropout)
        self.trg_embed = nn.Linear(self.in_feature_size, hidden_size)
        self.pe = PositionalEncoding(hidden_size, mask_count=True)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        if num_layers == 2:

            self.dec_spa_trans = TransformerDecoderLayer(
                size=hidden_size, ff_size=ff_size, num_heads=num_heads,
                dropout=dropout, decoder_trg_trg=decoder_trg_trg_)
            
            self.dec_tem_trans = TransformerDecoderLayer(
                size=hidden_size, ff_size=ff_size, num_heads=num_heads,
                dropout=dropout, decoder_trg_trg=decoder_trg_trg_)

            self.spa_pos_emb = nn.Parameter(torch.zeros(1, 3, hidden_size)) # [1, 3, C] for pose, rhand, lhand
            self.tem_pos_emb = nn.Parameter(torch.zeros(1, 8, hidden_size)) # [1, T, C]
            
            self.layer_norm_mid = nn.LayerNorm(hidden_size, eps=1e-6)
            self.output_layer_mid = nn.Linear(hidden_size, self.in_feature_size, bias=False)
            self.o1_embed = nn.Linear(trg_size, hidden_size)
            # self.o2_embed = nn.Linear((trg_size // 3) * 4, hidden_size)
            self.o2_embed = nn.Linear(trg_size, hidden_size)

            self.layers_mha_ac = TransformerDecoderLayer(
                size=hidden_size, ff_size=ff_size, num_heads=num_heads,
                dropout=dropout, decoder_trg_trg=decoder_trg_trg_)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

        # Output layer to be the size of joints vector + 1 for counter (total is trg_size)
        self.output_layer = nn.Linear(hidden_size, trg_size, bias=False)

        if freeze:
            freeze_params(self)

    def forward(self,
                t,
                trg_embed: Tensor = None,
                encoder_output: Tensor = None,
                src_mask: Tensor = None,
                trg_mask: Tensor = None,
                **kwargs):

        assert trg_mask is not None, "trg_mask required for Transformer"
        time_embed = self.time_mlp(t)[:, None, :].repeat(1, encoder_output.shape[1], 1)
        
        condition = encoder_output + time_embed
        condition = self.pos_drop(condition)
        
        B, _, _ = condition.shape
        B, H, T, N = trg_embed.shape
        
        spa_condition = einops.repeat(condition, 'b t h -> (b x) t h', x=T)
        tem_condition = einops.repeat(condition, 'b t h -> (b x) t h', x=N)
        spa_src_mask = einops.repeat(src_mask, 'b 1 h -> (b x) 1 h', x=T)
        tem_src_mask = einops.repeat(src_mask, 'b 1 h -> (b x) 1 h', x=N)
        
        spa_padding_mask = torch.ones(B*T, 1, N, N, dtype=torch.bool, device=trg_embed.device)
        tem_padding_mask = torch.ones(B*N, 1, T, T, dtype=torch.bool, device=trg_embed.device)
        # Create subsequent mask for decoding
        spa_sub_mask = subsequent_mask(N).type_as(trg_mask)
        tem_sub_mask = subsequent_mask(T).type_as(trg_mask)
        x = trg_embed
        # x = einops.rearrange(x, "b t n h -> (b t) n h")
        x = einops.rearrange(x, "b h t n -> (b t) n h")
        x = self.emb_dropout(x)
        x = x + self.spa_pos_emb
        x, h = self.dec_spa_trans(x=x, memory=spa_condition,
                             src_mask=spa_src_mask, trg_mask=spa_sub_mask, padding_mask=spa_padding_mask)

        # Temporal Transformer
        x = einops.rearrange(x, "(b t) n h -> (b n) t h", b=B)
        x = self.emb_dropout(x)
        x = x + self.tem_pos_emb
        x, h = self.dec_spa_trans(x=x, memory=tem_condition,
                             src_mask=tem_src_mask, trg_mask=tem_sub_mask, padding_mask=tem_padding_mask)

        x = einops.rearrange(x, "(b n) t h -> b t n h", b=B)
        
        x = self.layer_norm_mid(x)
        x = self.output_layer_mid(x)

        # Apply a layer normalisation
        x = self.layer_norm(x)
        # Output layer turns it back into vectors of size trg_size
        output = self.output_layer(x)
        output = einops.rearrange(output, "b t n h -> b h t n")
        return output

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, len(self.layers),
            self.layers[0].trg_trg_att.num_heads)
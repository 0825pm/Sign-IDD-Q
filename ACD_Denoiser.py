# coding: utf-8
import torch.nn as nn
import torch
import math
from torch import Tensor
import einops

from helpers import freeze_params, subsequent_mask
# transformer_layers.py는 수정할 필요가 없습니다.
from transformer_layers import PositionalEncoding, TransformerDecoderLayer

class SinusoidalPositionEmbeddings(nn.Module):
    """
    시간(timestep)을 입력받아 고차원 벡터로 변환하는 클래스
    """
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
    """
    개선된 조건 주입 방식(Separated Addition)과
    시공간 트랜스포머 버그가 수정된 최종 Denoiser 클래스
    """
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

        self.in_feature_size = trg_size
        self.out_feature_size = trg_size

        self.pos_drop = nn.Dropout(p=emb_dropout)
        self.trg_embed = nn.Linear(self.in_feature_size, hidden_size)
        self.pe = PositionalEncoding(hidden_size, mask_count=True)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        if num_layers == 2:
            # 공간(Spatial) 트랜스포머
            self.dec_spa_trans = TransformerDecoderLayer(
                size=hidden_size, ff_size=ff_size, num_heads=num_heads,
                dropout=dropout, decoder_trg_trg=decoder_trg_trg_)
            
            # 시간(Temporal) 트랜스포머
            self.dec_tem_trans = TransformerDecoderLayer(
                size=hidden_size, ff_size=ff_size, num_heads=num_heads,
                dropout=dropout, decoder_trg_trg=decoder_trg_trg_)

            # QAE의 압축된 토큰(N=3)과 프레임(T=8)을 위한 위치 임베딩
            self.spa_pos_emb = nn.Parameter(torch.zeros(1, 3, hidden_size)) # [1, 3, C]
            self.tem_pos_emb = nn.Parameter(torch.zeros(1, 8, hidden_size)) # [1, 8, C]
            
            self.layer_norm_mid = nn.LayerNorm(hidden_size, eps=1e-6)
            self.output_layer_mid = nn.Linear(hidden_size, self.in_feature_size, bias=False)
            self.o1_embed = nn.Linear(trg_size, hidden_size)
            self.o2_embed = nn.Linear(trg_size, hidden_size)

            self.layers_mha_ac = TransformerDecoderLayer(
                size=hidden_size, ff_size=ff_size, num_heads=num_heads,
                dropout=dropout, decoder_trg_trg=decoder_trg_trg_)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        # 시간(t) 임베딩을 위한 MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

        # 최종 출력 레이어
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
        
        # === 1. 조건 분리 ===
        
        # 시간 조건 (Time Condition): (B, H)
        time_embed = self.time_mlp(t)
        
        # 텍스트 조건 (Text Condition): (B, T_enc, H)
        # 원본과 달리 time_embed를 더하지 않습니다.
        text_condition = self.pos_drop(encoder_output)
        
        B, H, T, N = trg_embed.shape # (B, 512, 8, 3)
        
        # === 2. 각 트랜스포머에 맞게 조건 준비 ===
        
        # 2-1. 텍스트 조건 (Memory로 주입됨)
        # (B, T_enc, H) -> (B*T, T_enc, H)
        spa_text_cond = einops.repeat(text_condition, 'b t h -> (b x) t h', x=T)
        # (B, T_enc, H) -> (B*N, T_enc, H)
        tem_text_cond = einops.repeat(text_condition, 'b t h -> (b x) t h', x=N)
        # 소스 마스크도 동일하게 확장
        spa_src_mask = einops.repeat(src_mask, 'b 1 h -> (b x) 1 h', x=T)
        tem_src_mask = einops.repeat(src_mask, 'b 1 h -> (b x) 1 h', x=N)
        
        # 2-2. 시간 조건 (x에 직접 더해짐)
        # (B, H) -> (B*T, 1, H)
        time_embed_spa = einops.repeat(time_embed, 'b h -> (b t) 1 h', t=T)
        # (B, H) -> (B*N, 1, H)
        time_embed_tem = einops.repeat(time_embed, 'b h -> (b n) 1 h', n=N)

        # 타겟 마스크 (Self-Attention용)
        spa_padding_mask = torch.ones(B*T, 1, N, N, dtype=torch.bool, device=trg_embed.device)
        tem_padding_mask = torch.ones(B*N, 1, T, T, dtype=torch.bool, device=trg_embed.device)
        spa_sub_mask = subsequent_mask(N).type_as(trg_mask)
        tem_sub_mask = subsequent_mask(T).type_as(trg_mask)
        
        x = trg_embed # (B, H, T, N)
        
        # === 3. 공간(Spatial) 트랜스포머 ===
        # (B, H, T, N) -> (B*T, N, H)
        x_spa = einops.rearrange(x, "b h t n -> (b t) n h")
        x_spa = self.emb_dropout(x_spa)
        
        # (x) + (pos_emb) + (time_emb)
        x_spa = x_spa + self.spa_pos_emb + time_embed_spa # <-- 시간 조건 주입
        
        # memory에는 텍스트 조건만 주입
        x_spa, h = self.dec_spa_trans(x=x_spa, memory=spa_text_cond,
                                     src_mask=spa_src_mask, trg_mask=spa_sub_mask,
                                     padding_mask=spa_padding_mask)

        # === 4. 시간(Temporal) 트랜스포머 ===
        # (B*T, N, H) -> (B*N, T, H)
        x_tem = einops.rearrange(x_spa, "(b t) n h -> (b n) t h", b=B, t=T)
        x_tem = self.emb_dropout(x_tem)
        
        # (x) + (pos_emb) + (time_emb)
        x_tem = x_tem + self.tem_pos_emb + time_embed_tem # <-- 시간 조건 주입
        
        # [버그 수정] dec_tem_trans를 호출
        # memory에는 텍스트 조건만 주입
        x_tem, h = self.dec_tem_trans(x=x_tem, memory=tem_text_cond,
                                     src_mask=tem_src_mask, trg_mask=tem_sub_mask,
                                     padding_mask=tem_padding_mask)

        # === 5. 출력 ===
        # (B*N, T, H) -> (B, T, N, H)
        x_out = einops.rearrange(x_tem, "(b n) t h -> b t n h", b=B, n=N)
        
        x_out = self.layer_norm_mid(x_out)
        x_out = self.output_layer_mid(x_out)

        x_out = self.layer_norm(x_out)
        output = self.output_layer(x_out) # (B, T, N, H)
        
        # 원본 출력 형태 (B, H, T, N)로 복원
        output = einops.rearrange(output, "b t n h -> b h t n")
        return output
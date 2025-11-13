import torch
import torch.nn as nn
import math
import einops
from QFORMER import QFormer
    

class BertLayerNorm(nn.Module):
    """TF 스타일의 LayerNorm (epsilon이 제곱근 안에 있음)"""
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class Attention(nn.Module):
    """Standard Multi-head Attention"""
    def __init__(self, num_heads, size):
        super(Attention, self).__init__()
        assert size % num_heads == 0
        self.head_size = size // num_heads
        self.num_heads = num_heads
        self.k_layer = nn.Linear(size, size)
        self.v_layer = nn.Linear(size, size)
        self.q_layer = nn.Linear(size, size)
        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        batch_size = k.size(0)
        k = self.k_layer(k).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        v = self.v_layer(v).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        q = self.q_layer(q).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        
        q = q / math.sqrt(self.head_size)
        scores = torch.matmul(q, k.transpose(2, 3))
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        
        attention = self.softmax(scores)
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_size)
        return self.output_layer(context)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = BertLayerNorm(dim)
        self.attn = Attention(heads, dim)
        self.norm2 = BertLayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        x = x + self.ffn(self.norm2(x))
        return x

class Encoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(dim, heads, mlp_dim, dropout) for _ in range(depth)])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
            
        return x

# 논문에 명시된 Structured Prediction Layer (SPL)
# modules/sp_layer.py의 코드를 가져옴
SIGN_POSE_SKELETON = [
    [(-1, 0, "neck")],
    [(0, 1, "head"), (0, 5, "LeftUpArm"), (0, 2, "RightUpArm")],
    [(2, 3, "RightElbow"), (5, 6, "LeftElbow")],
    [(3, 4, "RightWrist"), (6, 7, "LeftWrist")]]

SIGN_HAND_SKELETON = [
    [(-1, 0, "Wrist")],
    [(0, 1, "hand1"), (0, 5, "hand5"), (0, 9, "hand9"), (0, 13, "hand13"), (0, 17, "hand17")],
    [(1, 2, "hand2"), (5, 6, "hand6"), (9, 10, "hand10"), (13, 14, "hand14"), (17, 18, "hand18")],
    [(2, 3, "hand3"), (6, 7, "hand7"), (10, 11, "hand11"), (14, 15, "hand15"), (18, 19, "hand19")],
    [(3, 4, "hand4"), (7, 8, "hand8"), (11, 12, "hand12"), (15, 16, "hand16"), (19, 20, "hand20")]]

class SP_block(nn.Module):
    def __init__(self, input_size, hid_size, out_size, L_num):
        super().__init__()
        layers = [nn.Linear(input_size, hid_size), nn.ReLU()]
        for _ in range(L_num - 1):
            layers.extend([nn.Linear(hid_size, hid_size), nn.ReLU()])
        layers.append(nn.Linear(hid_size, out_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class SPL(nn.Module):
    def __init__(self, input_size, hidden_layers, hidden_units, joint_size, SKELETON):
        super().__init__()
        self.input_size = input_size
        if SKELETON == "sign_pose":
            self.skeleton = SIGN_POSE_SKELETON
            self.num_joints = 8
        elif SKELETON == "sign_hand":
            self.skeleton = SIGN_HAND_SKELETON
            self.num_joints = 21
        else:
            raise ValueError(f"{SKELETON} is not a valid skeleton type!")
        
        kinematic_tree = {entry[1]: [([entry[0]] if entry[0] > -1 else []), entry[2]] for layer in self.skeleton for entry in layer}
        self.prediction_order = list(range(self.num_joints))
        self.indexed_skeleton = {i: [kinematic_tree[i][0], i, kinematic_tree[i][1]] for i in self.prediction_order}
        
        self.joint_predictions = nn.ModuleList()
        for joint_key in self.prediction_order:
            parent_ids, _, _ = self.indexed_skeleton[joint_key]
            current_input_size = self.input_size + joint_size * len(parent_ids)
            self.joint_predictions.append(SP_block(current_input_size, hidden_units, joint_size, hidden_layers))

    def forward(self, x):
        out = {}
        for joint_key in self.prediction_order:
            parent_ids, _, _ = self.indexed_skeleton[joint_key]
            parent_feats = [out[i] for i in parent_ids]
            x_input = torch.cat([x] + parent_feats, dim=-1) if parent_feats else x
            out[joint_key] = self.joint_predictions[joint_key](x_input)
        
        return torch.cat([out[i] for i in self.prediction_order], dim=-1)

class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., length=27):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.linear_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_1, x_2, x_3):
        B, N, C = x_1.shape
        B, N_1, C = x_3.shape

        q = self.linear_q(x_1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.linear_k(x_2).reshape(B, N_1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.linear_v(x_3).reshape(B, N_1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class QAE(nn.Module):
    def __init__(self, cfg):
        super(QAE, self).__init__()
        
        # 설정값
        embed_dim = cfg["model"]["qae"]["hidden_size"]
        depth = cfg["model"]["qae"]["depth"]
        num_heads = cfg["model"]["qae"]["num_heads"]
        hidden_dim = cfg["model"]["qae"]["ff_size"]
        self.num_tokens = cfg["model"]["qae"]["num_tokens"]
        
        drop_rate = cfg["model"]["qae"]["drop_rate"]
        attn_drop_rate = cfg["model"]["qae"]["attn_drop_rate"]
        qkv_bias = cfg["model"]["qae"]["qkv_bias"]
        qk_scale = cfg["model"]["qae"]["qk_scale"]
        # 1. 임베딩 및 Positional Encoding
        self.pose_emb = nn.Linear(8 * 3, embed_dim)
        self.rhand_emb = nn.Linear(21 * 3, embed_dim)
        self.lhand_emb = nn.Linear(21 * 3, embed_dim)

        self.spa_pos_emb = nn.Parameter(torch.zeros(1, 3, embed_dim)) # [1, 3, C] for pose, rhand, lhand
        self.tem_pos_emb = nn.Parameter(torch.zeros(1, cfg["data"]["max_sent_length"], embed_dim)) # [1, T, C]

        # 2. 인코더 (논문과 동일한 Transformer 구조)
        self.enc_spa_vit = Encoder(dim=embed_dim, depth=depth, heads=num_heads, mlp_dim=hidden_dim, dropout=0.1)
        self.enc_tem_vit = Encoder(dim=embed_dim, depth=depth, heads=num_heads, mlp_dim=hidden_dim, dropout=0.1)
        
        # --- VAE 변경: QFormer 출력을 mu와 log_var로 변환하는 레이어 추가 ---
        self.body_to_mu = nn.Linear(embed_dim, embed_dim)
        self.body_to_log_var = nn.Linear(embed_dim, embed_dim)
        
        self.rhand_to_mu = nn.Linear(embed_dim, embed_dim)
        self.rhand_to_log_var = nn.Linear(embed_dim, embed_dim)
        
        self.lhand_to_mu = nn.Linear(embed_dim, embed_dim)
        self.lhand_to_log_var = nn.Linear(embed_dim, embed_dim)
        # --- VAE 변경 끝 ---
        
        self.pose_query = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.pose_query.data.normal_(mean=0.0, std=0.02)
        self.pose_pos_emb = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.pose_pos_emb.data.normal_(mean=0.0, std=0.02)
        
        self.rhand_query = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.rhand_query.data.normal_(mean=0.0, std=0.02)
        self.rhand_pos_emb = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.rhand_pos_emb.data.normal_(mean=0.0, std=0.02)
        
        self.lhand_query = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.lhand_query.data.normal_(mean=0.0, std=0.02)
        self.lhand_pos_emb = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.lhand_pos_emb.data.normal_(mean=0.0, std=0.02)

        self.body_qformer = QFormer(embed_dim=embed_dim,
                               drop_rate=drop_rate,
                               depth=depth,
                               num_heads=num_heads
                               )      
        
        self.rhand_qformer = QFormer(embed_dim=embed_dim,
                               drop_rate=drop_rate,
                               depth=depth,
                               num_heads=num_heads
                               )      
        
        
        self.lhand_qformer = QFormer(embed_dim=embed_dim,
                               drop_rate=drop_rate,
                               depth=depth,
                               num_heads=num_heads
                               )
        
        # 4. 디코더 (논문과 동일한 Transformer 구조)
        self.dec_spa_vit = Encoder(dim=embed_dim, depth=depth, heads=num_heads, mlp_dim=hidden_dim, dropout=0.1)
        self.dec_tem_vit = Encoder(dim=embed_dim, depth=depth, heads=num_heads, mlp_dim=hidden_dim, dropout=0.1)
        self.dec_ca = Cross_Attention(embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop_rate, proj_drop=drop_rate)
        self.dec_token = nn.Parameter(torch.zeros(1, cfg["data"]["max_sent_length"], embed_dim))
        # 5. SPL (Structured Prediction Layer)
        self.pose_spl = SPL(input_size=embed_dim, hidden_layers=5, hidden_units=embed_dim, joint_size=3, SKELETON="sign_pose")
        self.hand_spl = SPL(input_size=embed_dim, hidden_layers=5, hidden_units=embed_dim, joint_size=3, SKELETON="sign_hand")        
    
    def _get_mask(self, x_len, size, device):
        pos = torch.arange(0, size, device=device).unsqueeze(0).repeat(x_len.size(0), 1)
        mask = pos < x_len.unsqueeze(1)
        return mask

    def reparameterize(self, mu, log_var):
        """VAE의 리파라미터화 트릭"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    # --- VAE 변경 끝 ---
    
    def encode_pose(self, pose_input, pose_length):
        B, T, N, C = pose_input.shape
        device = pose_input.device

        body_input = pose_input[:, :, :8, :].reshape(B, T, -1)
        rhand_input = pose_input[:, :, 8:29, :].reshape(B, T, -1)
        lhand_input = pose_input[:, :, 29:, :].reshape(B, T, -1)
        
        pose_emb = self.pose_emb(body_input).unsqueeze(2)   # [B, T, 1, C]
        rhand_emb = self.rhand_emb(rhand_input).unsqueeze(2) # [B, T, 1, C]
        lhand_emb = self.lhand_emb(lhand_input).unsqueeze(2) # [B, T, 1, C]
        
        points_feat = torch.cat([pose_emb, rhand_emb, lhand_emb], dim=2) # [B, T, 3, C]
        
        # Spatial Transformer
        points_feat = einops.rearrange(points_feat, "b t n h -> (b t) n h")
        points_feat = points_feat + self.spa_pos_emb
        points_feat = self.enc_spa_vit(points_feat, mask=None)

        # Temporal Transformer
        points_feat = einops.rearrange(points_feat, "(b t) n h -> (b n) t h", b=B)
        points_feat = points_feat + self.tem_pos_emb[:, :T, :]
        
        points_mask = self._get_mask(pose_length, T, device)
        points_mask_repeated = einops.repeat(points_mask, "b t -> (b n) 1 1 t", n=3)
        points_feat = self.enc_tem_vit(points_feat, mask=points_mask_repeated)
        
        points_feat = einops.rearrange(points_feat, "(b n) t h -> b h t n", b=B, n=3) # [B, C, T, 3]
        

        body_feat = points_feat[..., 0:1].squeeze(-1).permute(0, 2, 1).contiguous()
        rhand_feat = points_feat[..., 1:2].squeeze(-1).permute(0, 2, 1).contiguous()
        lhand_feat = points_feat[..., 2:3].squeeze(-1).permute(0, 2, 1).contiguous()
        return body_feat, rhand_feat, lhand_feat

    def encode_text(self, text_input, device):
        input_tokens = self.tokenizer(
                text_input,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).to(device)

        input_ids = input_tokens["input_ids"].to(device)
        attention_mask = input_tokens["attention_mask"].to(device)

        txt_logits = self.txt_model(input_ids=input_ids, attention_mask=attention_mask)[0]
        output = txt_logits[torch.arange(txt_logits.shape[0]), (input_ids == self.tokenizer.eos_token_id).long().argmax(dim=-1)]
        return self.lm_head(output)
    
    def decode(self, quantized_feat, pose_length):
        B, C, T, N_parts = quantized_feat.shape
        T = max(pose_length)
        device = quantized_feat.device
        
        points_feat = einops.rearrange(quantized_feat, "b h t n -> (b t) n h")
        points_feat = points_feat + self.spa_pos_emb
        points_feat = self.dec_spa_vit(points_feat, mask=None)
        
        points_feat = einops.rearrange(points_feat, "(b t) n h -> (b n) t h", b=B)
        dec_token = einops.repeat(self.dec_token, "() f c -> b f c", b = B*3)[:, :T, :]
        # x_token = self.sequence_pos_encoding(x_token)
        points_feat = dec_token + self.dec_ca(dec_token, points_feat, points_feat)
        points_feat = points_feat + self.tem_pos_emb[:, :T, :]
        
        # points_mask = self._get_mask(pose_length, T, device)
        points_mask = self._get_mask(pose_length, T, device)
        points_mask_repeated = einops.repeat(points_mask, "b t -> (b n) 1 1 t", n=3)
        points_feat = self.dec_tem_vit(points_feat, mask=points_mask_repeated)
        
        rec_feat = einops.rearrange(points_feat, "(b n) t h -> (b t) n h", b=B, n=N_parts)
        
        dec_pose_feat = rec_feat[:, 0, :]
        dec_rhand_feat = rec_feat[:, 1, :]
        dec_lhand_feat = rec_feat[:, 2, :]
        
        dec_pose = self.pose_spl(dec_pose_feat).view(B, T, -1)
        dec_rhand = self.hand_spl(dec_rhand_feat).view(B, T, -1)
        dec_lhand = self.hand_spl(dec_lhand_feat).view(B, T, -1)
        
        # 50 joints = 8 (pose) + 21 (rhand) + 21 (lhand)
        # Note: SPL output is (B*T, num_joints * 3). Need to reshape.
        # SPL 내부적으로 8*3=24, 21*3=63 차원으로 나옴.
        reconstructed_pose = torch.cat([dec_pose, dec_rhand, dec_lhand], dim=-1).view(B, T, 50, 3)

        return reconstructed_pose

    def qformer(self, body_feat, rhand_feat, lhand_feat):
        B, T, C = body_feat.shape
        device = body_feat.device
        pose_query = self.pose_query.expand(B, -1, -1)
        rhand_query = self.rhand_query.expand(B, -1, -1)
        lhand_query = self.lhand_query.expand(B, -1, -1)
        
        body_emb = self.body_qformer(body_feat, pose_query, self.pose_pos_emb)
        rhand_emb = self.rhand_qformer(rhand_feat, rhand_query, self.rhand_pos_emb)
        lhand_emb = self.lhand_qformer(lhand_feat, lhand_query, self.lhand_pos_emb)
        
        quantized_feat = torch.cat([body_emb.unsqueeze(-1), rhand_emb.unsqueeze(-1), lhand_emb.unsqueeze(-1)], dim=-1).permute(0, 2, 1, 3).contiguous()

        return quantized_feat, body_emb, rhand_emb, lhand_emb
        
    def forward(self, pose_input, text_input, text_ids, pose_target, pose_length, mask, pool_mask, test=False):
        B, T, N, C = pose_input.shape
        device = pose_input.device

        # Encode and Quantize
        body_feat, rhand_feat, lhand_feat = self.encode_pose(pose_input, pose_length)
        
        # Clustering
        quantized_feat, body_emb, rhand_emb, lhand_emb = self.qformer(body_feat, rhand_feat, lhand_feat)
        
        # text_feat = self.encode_text(text_input, device)
        # Decode
        pose_decoded = self.decode(quantized_feat, pose_length)
        # Reconstruction Loss
        # recon_loss = reconstruction_loss(pose_decoded, pose_input)
        recon_loss = nn.L1Loss()(pose_decoded, pose_input)
        
        # Dummy values for compatibility with your existing training loop
        pose_decoded_text = pose_decoded # 1단계에서는 text 조건이 없음
        # commit_loss = torch.tensor([0.0], device=pose_input.device)
        # contra_loss =  self.cal_clip_loss(quantized_feat.mean(-1).mean(-1), text_feat)
        contra_loss = torch.tensor([0.0], device=pose_input.device)
        len_loss = torch.tensor([0.0], device=pose_input.device)
        # len_loss = nn.L1Loss()(predicted_pose_length, pose_length.float()) * 0.01
        
        # 이것이 KLD Loss이며, 기존의 'latent_loss' 더미 값을 대체합니다.
        latent_loss = torch.tensor([0.0], device=device)
        indices = torch.tensor([[0]], device=device)
        commit_loss = torch.tensor([0.0], device=device)
        
        return pose_decoded, pose_decoded_text, recon_loss, commit_loss, contra_loss, len_loss, latent_loss, indices
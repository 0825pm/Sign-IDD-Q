import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

#
# 1. `models/diffusion/gaussian_diffusion.py`의 코드 (원본)
#
#

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = arr.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        return torch.linspace(1e-4, 0.02, num_diffusion_timesteps)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

class GaussianDiffusion(nn.Module):
    def __init__(self, betas, loss_type, model_mean_type, model_var_type):
        super().__init__()
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type

        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.alphas_cumprod_next = F.pad(self.alphas_cumprod[1:], (0, 1), value=0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_mean_variance(self, denoise_fn, x_t, t, clip_denoised, model_kwargs):
        pred_eps = denoise_fn(x_t, t, **model_kwargs)
        
        # fast streaming inference를 위해 history_kv를 반환하도록 수정
        history_kv = None
        if isinstance(pred_eps, tuple):
            pred_eps, history_kv = pred_eps
            
        x_start_pred = self._predict_xstart_from_eps(x_t, t, pred_eps)
        if clip_denoised:
            x_start_pred.clamp_(-1.0, 1.0) # VAE 잠재 벡터 범위에 맞게 조정 필요

        mean, variance, log_variance = self.q_posterior_mean_variance(x_start_pred, x_t, t)
        
        return mean, variance, log_variance, history_kv

    @torch.no_grad()
    def p_sample(self, denoise_fn, x_t, t, clip_denoised, model_kwargs):
        mean, _, log_variance, history_kv = self.p_mean_variance(
            denoise_fn, x_t, t, clip_denoised, model_kwargs
        )
        noise = torch.randn_like(x_t)
        if t[0] == 0:
            return mean, history_kv
        
        sample = mean + (0.5 * log_variance).exp() * noise
        return sample, history_kv

    @torch.no_grad()
    def p_sample_loop(self, denoise_fn, shape, model_kwargs):
        device = self.betas.device
        x_t = torch.randn(shape, device=device)
        history_kv = model_kwargs.get("history_kv", None)

        for i in reversed(range(self.num_timesteps)):
            t = torch.tensor([i] * shape[0], device=device)
            model_kwargs["history_kv"] = history_kv
            x_t, history_kv = self.p_sample(
                denoise_fn, x_t, t, clip_denoised=True, model_kwargs=model_kwargs
            )
        
        return x_t, history_kv

    def training_losses(self, denoise_fn, x_start, t, model_kwargs, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_t = self.q_sample(x_start, t, noise=noise)
        
        # history_kv=None (학습 시에는 사용 안 함)
        model_kwargs["history_kv"] = None 
        pred_eps = denoise_fn(x_t, t, **model_kwargs)

        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, pred_eps)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, pred_eps)
        else:
            raise NotImplementedError()

        return loss

#
# 2. `models/diffusion/respace.py`의 코드 (원본)
#
#

def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            steps = int(section_counts[len("ddim") :])
            return list(range(0, num_timesteps, num_timesteps // steps))
        raise ValueError(f"unknown timestep schedule: {section_counts}")
    size = num_timesteps // section_counts
    extra = num_timesteps % section_counts
    start = 0
    res = []
    for _ in range(section_counts):
        count = size + (1 if extra > 0 else 0)
        res.append(list(range(start, start + count)))
        start += count
        extra -= 1
    return res

class SpacedDiffusion(GaussianDiffusion):
    def __init__(self, betas, loss_type, model_mean_type, model_var_type, use_timesteps, respaced_seq=None):
        self.use_timesteps = use_timesteps
        self.timestep_map = list(range(betas.shape[0]))
        self.original_num_steps = betas.shape[0]

        if respaced_seq is not None:
            betas = self.get_betas_from_respaced_seq(betas, respaced_seq)
        
        super().__init__(betas, loss_type, model_mean_type, model_var_type)

    def get_betas_from_respaced_seq(self, betas, respaced_seq):
        print(f"Using respaced sequence: {len(respaced_seq)} steps")
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        new_alphas_cumprod = alphas_cumprod[respaced_seq]
        new_alphas_cumprod_prev = F.pad(new_alphas_cumprod[:-1], (1, 0), value=1.0)
        
        new_betas = 1.0 - (new_alphas_cumprod / new_alphas_cumprod_prev)
        
        self.timestep_map = respaced_seq
        return new_betas

    def p_sample_loop(self, denoise_fn, shape, model_kwargs):
        device = self.betas.device
        x_t = torch.randn(shape, device=device)
        history_kv = model_kwargs.get("history_kv", None)

        for i in reversed(range(self.num_timesteps)):
            t = torch.tensor([i] * shape[0], device=device)
            mapped_t = torch.tensor([self.timestep_map[i]] * shape[0], device=device)
            
            model_kwargs["history_kv"] = history_kv
            x_t, history_kv = self.p_sample(
                denoise_fn, x_t, t, clip_denoised=True, model_kwargs=model_kwargs
            )
        
        return {'sample': x_t, 'history_kv': history_kv} # history_kv 반환

    def training_losses(self, denoise_fn, x_start, t, model_kwargs, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        t_mapped = torch.tensor([self.timestep_map[i] for i in t], device=t.device)
        x_t = self.q_sample(x_start, t, noise=noise)
        
        model_kwargs["history_kv"] = None
        pred_eps = denoise_fn(x_t, t_mapped, **model_kwargs)

        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, pred_eps)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, pred_eps)
        else:
            raise NotImplementedError()
        return loss

#
# 3. `models/llama_model.py`의 코드 (원본)
#
#

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class Attention(nn.Module):
    def __init__(self, n_heads: int, n_kv_heads: int, dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, history_kv: tuple, mask: torch.Tensor):
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        if history_kv is not None:
            prev_k, prev_v = history_kv
            prev_k = prev_k.to(xq.device)
            prev_v = prev_v.to(xq.device)
            xk = torch.cat([prev_k, xk], dim=1)
            xv = torch.cat([prev_v, xv], dim=1)
        
        current_kv = (xk, xv)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        output = torch.matmul(scores, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        return self.wo(output), current_kv

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, n_heads: int, n_kv_heads: int, dim: int, hidden_dim: int):
        super().__init__()
        self.attention = Attention(n_heads, n_kv_heads, dim)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor, history_kv: tuple, mask: torch.Tensor):
        h, current_kv = self.attention(self.attention_norm(x), history_kv, mask)
        h = x + h
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, current_kv

class Llama(nn.Module):
    def __init__(self, dim_model, n_layers, n_heads, n_kv_heads, hidden_dim, text_embed_dim, latent_dim):
        super().__init__()
        self.n_layers = n_layers
        self.dim_model = dim_model
        
        self.layers = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(TransformerBlock(n_heads, n_kv_heads, dim_model, hidden_dim))
        
        self.norm = RMSNorm(dim_model)
        self.output = nn.Linear(dim_model, latent_dim)
        
        self.time_embed = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.SiLU(),
            nn.Linear(dim_model, dim_model),
        )
        
        # QFormer 토큰(x_t) 임베딩
        self.latent_embed = nn.Linear(latent_dim, dim_model) 
        
        # 조건 (히스토리 토큰 + 텍스트) 임베딩
        # (히스토리는 C_latent, 텍스트는 C_text이므로, wrapper에서 C_latent로 통일 필요)
        self.cond_embed = nn.Linear(dim_model, dim_model) # 입력 차원을 dim_model로 가정

    def _timestep_embedding(self, t, dim):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        if dim % 2 == 0:
            return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        else:
            return torch.cat([torch.cos(args), torch.sin(args), torch.zeros_like(args[:, :1])], dim=-1)

    def forward(self, tokens, t, cond_embed=None, history_kv=None, motion_mask_cond=None, x_t_mask=None):
        # tokens: (B, 1, C_latent) - 노이즈가 낀 현재 예측 토큰 (x_t)
        # cond_embed: (B, T_hist, C_model) - 히스토리 토큰 + 텍스트 조건
        
        bsz, seqlen = tokens.shape[0], tokens.shape[1]
        
        h = self.latent_embed(tokens)
        t_emb = self.time_embed(self._timestep_embedding(t, self.dim_model))
        t_emb = t_emb.unsqueeze(1)

        if cond_embed is not None:
            h_cond = self.cond_embed(cond_embed)
            h = torch.cat((h_cond, t_emb, h), dim=1)
        else:
            h = torch.cat((t_emb, h), dim=1)

        mask = None
        if seqlen > 1 or (cond_embed is not None):
            mask = torch.full((1, 1, h.shape[1], h.shape[1]), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            
            # Causal 마스크: [history, time, x_t]
            # x_t (1토큰)는 history와 time을 모두 참조
            # history (T_hist 토큰)는 history 내에서만 causal 참조
            # time (1토큰)는 history만 참조
            
            # MotionStreamer의 마스크 로직 (단순화된 Causal Mask)
            # 여기서는 Llama의 기본 Causal 마스크를 사용
            if motion_mask_cond is not None:
                 mask[..., -seqlen:, :-seqlen] = motion_mask_cond

        if history_kv is None:
            history_kv = [None] * self.n_layers
            
        current_kv_list = []
        for i, layer in enumerate(self.layers):
            h, current_kv = layer(h, history_kv[i], mask)
            current_kv_list.append(current_kv)

        h = self.norm(h)
        output = self.output(h[:, -seqlen:, :]) # x_t에 해당하는 마지막 1개 토큰의 출력만 반환

        if history_kv[0] is not None:
            return output, current_kv_list
        else:
            return output

#
# 4. `models/diffusion/diffusion_utils.py`의 헬퍼 함수
#
#

def create_qformer_diffusion(
    timestep_respacing,
    loss_type='l1',
    noise_schedule="linear",
    num_diffusion_timesteps=1000
):
    betas = get_named_beta_schedule(noise_schedule, num_diffusion_timesteps)
    
    if not timestep_respacing:
        timestep_respacing = [num_diffusion_timesteps]

    return SpacedDiffusion(
        betas=betas,
        loss_type=loss_type,
        model_mean_type='eps', # 'eps'
        model_var_type='fixed_small', # 'fixed_small'
        use_timesteps=space_timesteps(num_diffusion_timesteps, timestep_respacing),
        respaced_seq=timestep_respacing
    )

#
# 5. QFormer 토큰 생성을 위한 SignStreamer (최종 모델)
#

class SignStreamerQFormer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # 1. QAE 설정과 일치 필요
        self.latent_dim = cfg["model"]["qae"]["hidden_size"] 
        self.num_qformer_tokens = cfg["model"]["qae"]["num_tokens"]
        self.num_parts = 3 # body, rhand, lhand
        self.num_stream_tokens = self.num_qformer_tokens * self.num_parts
        
        # 2. Streamer (Llama) 설정
        llama_cfg = cfg["model"]["streamer"]
        self.dim_model = llama_cfg["dim_model"]
        self.text_embed_dim = llama_cfg["text_embed_dim"] # QAE 텍스트 인코더 출력 차원

        # 3. Diffusion 설정
        diff_cfg = cfg["model"]["diffusion"]
        self.diffusion = create_qformer_diffusion(
            timestep_respacing=diff_cfg["timestep_respacing"], # 예: "ddim100"
            loss_type=diff_cfg["loss_type"],
            num_diffusion_timesteps=diff_cfg["num_timesteps"]
        )
        
        # 4. Llama 모델
        # (텍스트 임베딩 차원과 히스토리(latent) 임베딩 차원이 다를 수 있음)
        # Llama 모델은 입력(x_t)으로 C_latent(latent_dim)를 받고,
        # 조건(cond_embed)으로 C_model(dim_model)을 받음
        self.llama = Llama(
            dim_model=self.dim_model,
            n_layers=llama_cfg["n_layers"],
            n_heads=llama_cfg["n_heads"],
            n_kv_heads=llama_cfg["n_kv_heads"],
            hidden_dim=llama_cfg["hidden_dim"],
            text_embed_dim=self.dim_model, # Llama 내부 cond_embed 차원
            latent_dim=self.latent_dim     # Llama 내부 x_t(latent) 임베딩 차원
        )
        
        # 5. 프로젝터 (차원 통일)
        # (B, C_text) -> (B, C_model)
        self.text_projector = nn.Linear(self.text_embed_dim, self.dim_model)
        # (B, C_latent) -> (B, C_model)
        self.history_projector = nn.Linear(self.latent_dim, self.dim_model)


    def forward(self, z_sequence, text_condition):
        """학습 (Training)"""
        # z_sequence: (B, N_stream, C_latent) - QAE의 mu값들
        # text_condition: (B, C_text) - QAE의 텍스트 인코더 출력
        
        B, N_stream, C_latent = z_sequence.shape
        device = z_sequence.device
        
        # 1. 예측할 토큰 인덱스(i) 랜덤 선택 (1 ~ N_stream-1)
        i = torch.randint(1, N_stream, (B,), device=device)
        
        # 2. 타겟(x_start)과 히스토리 분리
        # (B, 1, C_latent)
        z_target = z_sequence[torch.arange(B), i].unsqueeze(1) 
        
        # (B, i, C_latent) - 패딩 필요
        max_hist_len = i.max()
        z_history = torch.zeros((B, max_hist_len, C_latent), device=device)
        for b in range(B):
            hist_len = i[b]
            z_history[b, :hist_len] = z_sequence[b, :hist_len]
            
        # 3. 조건 임베딩 (프로젝션)
        # (B, 1, C_model)
        text_proj = self.text_projector(text_condition).unsqueeze(1)
        # (B, i, C_model)
        history_proj = self.history_projector(z_history)
        
        # (B, i+1, C_model)
        full_cond = torch.cat([history_proj, text_proj], dim=1)

        # 4. 조건 마스크
        # (B, 1, i+1) - x_t가 full_cond를 모두 참조
        mask = torch.zeros((B, 1, max_hist_len + 1), device=device)
        # 배치별 패딩 마스크 적용
        for b in range(B):
             mask[b, :, i[b]:] = float("-inf") # 텍스트(i[b]) 이후 패딩된 히스토리
        
        # 5. 확산 타임스텝(k) 랜덤 선택 (0 ~ T-1)
        k_t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=device)
        
        # 6. 확산 손실 계산
        model_kwargs = {
            "cond_embed": full_cond, 
            "motion_mask_cond": mask
        }
        
        loss = self.diffusion.training_losses(
            self.llama, 
            z_target, 
            k_t, 
            model_kwargs=model_kwargs
        )
        return loss

    @torch.no_grad()
    def generate(self, text_condition):
        """생성 (Inference) - QFormer 토큰 시퀀스 생성"""
        B = text_condition.shape[0]
        device = text_condition.device
        
        # (B, 1, C_model)
        text_proj = self.text_projector(text_condition).unsqueeze(1)
        
        # 생성된 토큰(C_latent)을 저장할 리스트
        z_generated_latent = [] 
        
        history_kv = None # Llama의 fast inference state
        
        # 생성된 토큰(C_model)을 히스토리로 사용
        z_history_proj = torch.zeros((B, 0, self.dim_model), device=device)

        for i in range(self.num_stream_tokens):
            
            # 1. 조건 준비 (이전 히스토리 + 텍스트)
            # (B, i+1, C_model)
            cond = torch.cat([z_history_proj, text_proj], dim=1)
            
            # 2. 마스크 (현재 토큰이 모든 조건을 참조)
            mask = torch.zeros((B, 1, i + 1), device=device)
            
            # 3. 샘플링할 노이즈 shape
            shape = (B, 1, self.latent_dim)
            
            model_kwargs = {
                "cond_embed": cond, 
                "motion_mask_cond": mask, 
                "history_kv": history_kv
            }
            
            # 4. 다음 토큰(z_next) 샘플링 (DDIM/DDPM)
            # (SpacedDiffusion.p_sample_loop가 history_kv를 반환하도록 수정됨)
            sample_out = self.diffusion.p_sample_loop(
                self.llama, 
                shape, 
                model_kwargs=model_kwargs
            )
            
            # (B, 1, C_latent)
            z_next_latent = sample_out['sample']
            history_kv = sample_out['history_kv']
            
            # 5. 생성된 z_next를 히스토리에 추가
            z_generated_latent.append(z_next_latent)
            
            # 다음 스텝의 히스토리 조건으로 사용하기 위해 프로젝션
            # (B, 1, C_model)
            z_next_proj = self.history_projector(z_next_latent)
            z_history_proj = torch.cat([z_history_proj, z_next_proj], dim=1)
            
        # (B, N_stream, C_latent)
        z_sequence = torch.cat(z_generated_latent, dim=1)
        
        # QAE.py의 형식에 맞게 분리
        # (B, num_tokens, C_latent)
        body_mu = z_sequence[:, :self.num_qformer_tokens, :]
        rhand_mu = z_sequence[:, self.num_qformer_tokens : self.num_qformer_tokens * 2, :]
        lhand_mu = z_sequence[:, self.num_qformer_tokens * 2 :, :]
        
        return body_mu, rhand_mu, lhand_mu
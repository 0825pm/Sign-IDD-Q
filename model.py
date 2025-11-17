# coding: utf-8
import torch
import torch.nn as nn

from torch import Tensor
import einops
from encoder import Encoder
from ACD import ACD
from QAE import QAE
from batch import Batch
from embeddings import Embeddings
from vocabulary import Vocabulary
from initialization import initialize_model
from constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN, TARGET_PAD
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self,cfg: dict, 
                 encoder: Encoder, 
                 ACD: ACD,
                 QAE: QAE,
                 src_embed: Embeddings, 
                 src_vocab: Vocabulary, 
                 trg_vocab: Vocabulary, 
                 in_trg_size: int, 
                 out_trg_size: int,
                 ):
        """
        Create Sign-IDD

        :param encoder: encoder
        :param ACD: ACD
        :param src_embed: source embedding
        :param trg_embed: target embedding
        :param src_vocab: source vocabulary
        :param trg_vocab: target vocabulary
        """
        super(Model, self).__init__()

        model_cfg = cfg["model"]
        self.src_embed = src_embed
        self.encoder = encoder
        self.ACD = ACD
        self.QAE = QAE
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.bos_index = self.src_vocab.stoi[BOS_TOKEN]
        self.pad_index = self.src_vocab.stoi[PAD_TOKEN]
        self.eos_index = self.src_vocab.stoi[EOS_TOKEN]
        self.target_pad = TARGET_PAD
        self.num_tokens = cfg["model"]["qae"]["num_tokens"]

        self.use_cuda = cfg["training"]["use_cuda"]

        self.in_trg_size = in_trg_size
        self.out_trg_size = out_trg_size
        
        self.pretrain = cfg["training"]["pretrain"]

        if not self.pretrain:
            for name, param in self.QAE.named_parameters():
                param.requires_grad = False
            self.QAE.eval()

        
    def forward(self, is_train: bool, src: Tensor, trg_input: Tensor, src_mask: Tensor, src_lengths: Tensor, trg_mask: Tensor):

        """
        First encodes the source sentence.
        Then produces the target one word at a time.

        :param src: source input
        :param trg_input: target input
        :param src_mask: source mask
        :param src_lengths: length of source inputs
        :param trg_mask: target mask
        :param trg_lengths: length of target inputs
        :return: diffusion_output
        """
        
        if len(trg_input.shape) == 3:
            trg_input = einops.rearrange(trg_input, "b t (j h) -> b t j h", h=3)
        
        if self.pretrain:
            body_feat, rhand_feat, lhand_feat = self.QAE.encode_pose(pose_input=trg_input,
                                                  pose_length=trg_mask[...,0].sum(dim=-1).ravel())
            qae_encoder_output, body_mu, body_log_var, rhand_mu, rhand_log_var, lhand_mu, lhand_log_var = self.QAE.qformer(body_feat, rhand_feat, lhand_feat)
            # qae_encoder_output, "b h t n -> b (t n) h")
            pose_output = self.QAE.decode(qae_encoder_output, trg_mask[...,0].sum(dim=-1).ravel())
            pose_output = einops.rearrange(pose_output, "b t j h -> b t (j h)")
            return pose_output, body_mu, body_log_var, rhand_mu, rhand_log_var, lhand_mu, lhand_log_var
        else:
            with torch.no_grad():
                body_feat, rhand_feat, lhand_feat = self.QAE.encode_pose(pose_input=trg_input,
                                                  pose_length=trg_mask[...,0].sum(dim=-1).ravel())
                qae_encoder_output, body_mu, body_log_var, rhand_mu, rhand_log_var, lhand_mu, lhand_log_var = self.QAE.qformer(body_feat, rhand_feat, lhand_feat)
                
        encoder_output = self.encode(src=src,
                                     src_length=src_lengths,
                                     src_mask=src_mask)
        
        # Diffusion the target sequence
        latent_target = qae_encoder_output
        # trg_input = einops.rearrange(trg_input, "b h t j -> b (t j) h")
        B, H, T, J = trg_input.shape
        diff_trg_mask = torch.ones(B, 1, T*J, T*J, dtype=torch.bool, device=trg_input.device)
        # trg_mask = None
                                
        diffusion_output = self.diffusion(is_train=is_train,
                                          encoder_output=encoder_output,
                                          trg_input=latent_target, # Original
                                          src_mask=src_mask,
                                          trg_mask=diff_trg_mask)
        
        
        # with torch.no_grad():
        pose_output = self.QAE.decode(diffusion_output, trg_mask[...,0].sum(dim=-1).ravel())
        pose_output = einops.rearrange(pose_output, "b t j h -> b t (j h)")

        return pose_output, body_mu, body_log_var, rhand_mu, rhand_log_var, lhand_mu, lhand_log_var

    def encode(self, src: Tensor, src_length: Tensor, src_mask: Tensor):

        """
        Encodes the source sentence.

        :param src:
        :param src_length:
        :param src_mask:
        :return: encoder outputs
        """

        # Encode an embedded source
        encode_output = self.encoder(embed_src=self.src_embed(src), 
                                     src_length=src_length, 
                                     mask=src_mask)

        return encode_output
    
    def diffusion(self, is_train: bool, encoder_output: Tensor, src_mask: Tensor, trg_input: Tensor, trg_mask: Tensor):
        
        """
        diffusion the target sentence.

        :param src: param encoder_output: encoder states for attention computation
        :param src_mask: source mask, 1 at valid tokens
        :param trg_input: target inputs
        :param trg_mask: mask for target steps
        :return: diffusion outputs
        """

        diffusion_output = self.ACD(is_train=is_train,
                                    encoder_output=encoder_output,
                                    input_3d=trg_input,
                                    src_mask=src_mask, 
                                    trg_mask=trg_mask)
        
        return diffusion_output
    
    def get_loss_for_batch(self, is_train, batch: Batch, loss_function: nn.Module) -> Tensor:
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param loss_function: loss function, computes for input and target
            a scalar loss for the complete batch
        :return: batch_loss: sum of losses over non-pad elements in the batch
        """
        # Forward through the batch input
        # get_loss_for_batch는 학습 중에만 호출되므로 is_train=True로
        # forward를 호출하여 3개의 텐서를 받음
        skel_out = self.forward(src=batch.src,
                                trg_input=batch.trg_input[:, :, :150],
                                src_mask=batch.src_mask,
                                src_lengths=self.num_tokens,
                                trg_mask=batch.trg_mask,
                                is_train=True) # 학습 모드로 강제

        # 'pretrain' 플래그 분기 제거
        
        # diffusion_pred, latent_target, pose_recon = skel_out
        diffusion_pred, body_mu, body_log_var, rhand_mu, rhand_log_var, lhand_mu, lhand_log_var = skel_out

        # E2E 손실 계산
        # 1. 확산 손실 (Diffusion Loss)
        # loss_diffusion = F.l1_loss(diffusion_pred.contiguous(), latent_target.contiguous())

        # 2. 재구성 손실 (Reconstruction Loss)
        #    loss_function은 training.py에서 전달된 self.loss (즉, Loss 클래스 인스턴스)임
        loss_recon = loss_function(diffusion_pred, batch.trg_input[:, :, :150])
        
        kld_body = -0.5 * torch.sum(1 + body_log_var - body_mu.pow(2) - body_log_var.exp(), dim=[1, 2]).mean()
        kld_rhand = -0.5 * torch.sum(1 + rhand_log_var - rhand_mu.pow(2) - rhand_log_var.exp(), dim=[1, 2]).mean()
        kld_lhand = -0.5 * torch.sum(1 + lhand_log_var - lhand_mu.pow(2) - lhand_log_var.exp(), dim=[1, 2]).mean()
        
        latent_loss = kld_body + kld_rhand + kld_lhand

        # print(f"DEBUG E2E Loss --> Diff: {loss_diffusion.item():.6f} | Recon: {loss_recon.item():.6f} (Weight: {self.recon_loss_weight})")
        
        # 3. 두 손실을 가중합
        batch_loss = loss_recon + latent_loss * 1e-5

        return batch_loss

def build_model(cfg: dict, src_vocab: Vocabulary, trg_vocab: Vocabulary):

    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param src_vocab: source vocabulary
    :param trg_vocab: target vocabulary
    :return: built and initialized model
    """
    full_cfg = cfg
    cfg = cfg["model"]

    src_padding_idx = src_vocab.stoi[PAD_TOKEN]
    trg_padding_idx = 0

    # Input target size is the joint vector length plus one for counter
    in_trg_size = cfg["trg_size"]
    # Output target size is the joint vector length plus one for counter
    out_trg_size = cfg["trg_size"]

    # Define source embedding
    src_embed = Embeddings(
        **cfg["encoder"]["embeddings"], vocab_size=len(src_vocab),
        padding_idx=src_padding_idx)
    
    ## Encoder -------
    enc_dropout = cfg["encoder"].get("dropout", 0.) # Dropout
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    assert cfg["encoder"]["embeddings"]["embedding_dim"] == \
           cfg["encoder"]["hidden_size"], \
           "for transformer, emb_size must be hidden_size"
    
    # Qformer AutoEncoder
    qae = QAE(cfg=full_cfg)
    
    # Transformer Encoder
    encoder = Encoder(**cfg["encoder"],
                      emb_size=src_embed.embedding_dim,
                      emb_dropout=enc_emb_dropout)
    
    # ACD
    diffusion = ACD(args=cfg, 
                    trg_vocab=trg_vocab)
    
    # Define the model
    model = Model(encoder=encoder,
                  ACD=diffusion,
                  QAE=qae,
                  src_embed=src_embed,
                  src_vocab=src_vocab,
                  trg_vocab=trg_vocab,
                  cfg=full_cfg,
                  in_trg_size=in_trg_size,
                  out_trg_size=out_trg_size)

    # Custom initialization of model parameters
    initialize_model(model, cfg, src_padding_idx, trg_padding_idx)

    return model
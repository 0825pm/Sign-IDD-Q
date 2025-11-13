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
            qae_encoder_output, body_emb, rhand_emb, lhand_emb = self.QAE.qformer(body_feat, rhand_feat, lhand_feat)
            # qae_encoder_output, "b h t n -> b (t n) h")
            pose_output = self.QAE.decode(qae_encoder_output, trg_mask[...,0].sum(dim=-1).ravel())
            pose_output = einops.rearrange(pose_output, "b t j h -> b t (j h)")
            return (pose_output, body_emb, rhand_emb, lhand_emb)
        else:
            with torch.no_grad():
                body_feat, rhand_feat, lhand_feat = self.QAE.encode_pose(pose_input=trg_input,
                                                  pose_length=trg_mask[...,0].sum(dim=-1).ravel())
                qae_encoder_output, body_emb, rhand_emb, lhand_emb = self.QAE.qformer(body_feat, rhand_feat, lhand_feat)
        # Encode the source sequence
        encoder_output = self.encode(src=src,
                                     src_length=src_lengths,
                                     src_mask=src_mask)
        
        # Diffusion the target sequence
        trg_input = qae_encoder_output.detach().clone()
        # trg_input = einops.rearrange(trg_input, "b h t j -> b (t j) h")
        B, H, T, J = trg_input.shape
        trg_mask = torch.ones(B, 1, T*J, T*J, dtype=torch.bool, device=trg_input.device)
        # trg_mask = None
                                
        diffusion_output = self.diffusion(is_train=is_train,
                                          encoder_output=encoder_output,
                                          trg_input=trg_input, # Original
                                          src_mask=src_mask,
                                          trg_mask=trg_mask)

        return (diffusion_output, trg_input)

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
        skel_out = self.forward(src=batch.src,
                                trg_input=batch.trg_input[:, :, :150],
                                src_mask=batch.src_mask,
                                # src_lengths=batch.src_lengths, # Original
                                src_lengths=self.num_tokens, # QAE
                                trg_mask=batch.trg_mask,
                                is_train=is_train)

        # compute batch loss using skel_out and the batch target
        if self.pretrain:
            pose_output, body_emb, rhand_emb, lhand_emb = skel_out
            batch_loss = loss_function(pose_output, batch.trg_input[:, :, :150])
        else:
            batch_loss = loss_function(skel_out[0], skel_out[1])

        # return batch loss = sum over all elements in batch that are not pad
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
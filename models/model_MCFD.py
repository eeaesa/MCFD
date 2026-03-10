import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional
from random import random
import numpy as np

class NLLSurvLoss(nn.Module):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    alpha: float
        TODO: document
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    """
    def __init__(self, alpha=0.0, eps=1e-7, reduction='sum'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def __call__(self, h, y, t, c):
        """
        Parameters
        ----------
        h: (n_batches, n_classes)
            The neural network output discrete survival predictions such that hazards = sigmoid(h).
        y_c: (n_batches, 2) or (n_batches, 3)
            The true time bin label (first column) and censorship indicator (second column).
        """

        return nll_loss(h=h, y=y.unsqueeze(dim=1), c=c.unsqueeze(dim=1),
                        alpha=self.alpha, eps=self.eps,
                        reduction=self.reduction)


# TODO: document better and clean up
def nll_loss(h, y, c, alpha=0.0, eps=1e-7, reduction='sum'):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    h: (n_batches, n_classes)
        The neural network output discrete survival predictions such that hazards = sigmoid(h).
    y: (n_batches, 1)
        The true time bin index label.
    c: (n_batches, 1)
        The censoring status indicator.
    alpha: float
        The weight on uncensored loss
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    References
    ----------
    Zadeh, S.G. and Schmid, M., 2020. Bias in cross-entropy-based training of deep survival networks. IEEE transactions on pattern analysis and machine intelligence.
    """
    # print("h shape", h.shape)

    # make sure these are ints
    y = y.type(torch.int64)
    c = c.type(torch.int64)

    hazards = torch.sigmoid(h)  # hazard function
    # print("hazards shape", hazards.shape)

    S = torch.cumprod(1 - hazards, dim=1)
    # print("S.shape", S.shape, S)

    S_padded = torch.cat([torch.ones_like(c), S], 1)
    # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # hazards[y] = hazards(1)
    # S[1] = S(1)

    # print("S_padded.shape", S_padded.shape, S_padded)

    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y + 1).clamp(min=eps)
    # print('s_prev.s_prev', s_prev.shape, s_prev)
    # print('h_this.shape', h_this.shape, h_this)
    # print('s_this.shape', s_this.shape, s_this)

    # c = 1 means censored. Weight 0 in this case
    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - c * torch.log(s_this)

    # print('uncensored_loss.shape', uncensored_loss.shape)
    # print('censored_loss.shape', censored_loss.shape)

    neg_l = censored_loss + uncensored_loss
    if alpha is not None:
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))

    return loss



def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))

def MLP_Block(dim1, dim2, dropout=0.1):
    r"""
    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    return nn.Sequential(
            nn.Linear(dim1, dim1//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim1//2, dim2), nn.ReLU(), nn.Dropout(dropout)
        )

#########################################
# FeatureDistillation
#########################################

def default_check(val, d):
    return val if exists_check(val) else d


def exists_check(val):
    return val is not None


class SELUGated(nn.Module):
    def forward(self, x):
        if x.shape[-1] % 2 == 0:
            a, b = x.chunk(2, dim=-1)
            return a * F.selu(b)
        return F.selu(x)


class GELUGated(nn.Module):
    def forward(self, x):
        if x.shape[-1] % 2 == 0:
            a, b = x.chunk(2, dim=-1)
            return a * F.gelu(b)
        return F.gelu(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

        self.has_context_norm = context_dim is not None
        self.norm_context = nn.LayerNorm(context_dim) if self.has_context_norm else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if self.has_context_norm:
            context = kwargs.get("context", None)
            if context is not None:
                kwargs["context"] = self.norm_context(context)

        return self.fn(x, **kwargs)


class MHSAtt(nn.Module):
    def __init__(
            self,
            query_dim,
            context_dim=None,
            heads=8,
            dropout=0.,
            temperature: float = 1.0,
            use_leaky_out: bool = False,
            use_gate: bool = True
    ):
        super().__init__()

        dim_head = query_dim // heads
        inner_dim = dim_head * heads
        context_dim = default_check(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head
        self.temperature = temperature
        self.scale = dim_head ** -0.5
        self.use_gate = use_gate

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)

        if use_leaky_out:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, query_dim),
                nn.LeakyReLU(negative_slope=1e-2)
            )
        else:
            self.to_out = nn.Linear(inner_dim, query_dim)

        self.gate = nn.Parameter(torch.ones(1)) if use_gate else None
        self.attn_weights = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, context=None, mask=None):
        """
        x: (B, N, C)
        context: (B, M, C)
        """
        B, N, _ = x.shape
        context = default_check(context, x)

        q = self.to_q(x)  # (B, N, H*D)
        kv = self.to_kv(context)  # (B, M, 2*H*D)
        k, v = kv.chunk(2, dim=-1)

        q = q.view(B, N, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(B, -1, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(B, -1, self.heads, self.dim_head).transpose(1, 2)
        # q/k/v: (B, H, N(or M), D)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # attn: (B, H, N, M)

        if exists_check(mask):
            # mask: (B, M)
            mask = mask[:, None, None, :]  # (B,1,1,M)
            max_neg = -torch.finfo(attn.dtype).max
            attn = attn.masked_fill(~mask, max_neg)

        attn = attn - attn.amax(dim=-1, keepdim=True)
        attn = (attn / self.temperature).softmax(dim=-1)
        self.attn_weights = attn

        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, N, D)

        out = out.transpose(1, 2).reshape(B, N, self.heads * self.dim_head)

        out = self.to_out(out)
        if self.use_gate:
            out = self.gate * out

        return out


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0., snn: bool = False):
        super().__init__()
        activation = SELUGated() if snn else GELUGated()
        hidden_dim = hidden_dim or dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            activation,
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class FeatureDistillation_module(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        out_dims: int,
        n_modalities: int = 2,
        latent_num: int = 64,
        latent_dim: int = 256,
        depth: int = 2,
        cross_head: int = 4,
        latent_heads: int = 4,
        attn_dropout: float = 0.455,
        ff_dropout: float = 0.364,
        weight_tie_layers: bool = False,
        self_per_cross_attn: int = 1,
        snn: bool = True,
        temperature: float = 1.0,
        use_leaky_out: bool = False,
        use_gate: bool = True,
        proj_dropout: float = 0.0,
    ):
        super().__init__()

        self.n_modalities = n_modalities
        self.latent_num = latent_num
        self.latent_dim = latent_dim
        self.depth = depth
        self.self_per_cross_attn = self_per_cross_attn

        # ----------------------
        # Latent tokens
        # ----------------------
        self.latents = nn.Parameter(torch.randn(latent_num, latent_dim) * 0.02)

        # ----------------------
        # Shared projection for modality features
        # ----------------------
        self.shared_proj = nn.Sequential(
            nn.Linear(embed_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Dropout(proj_dropout)
        )

        # ----------------------
        # Define single-layer factories (weight sharing aware)
        # ----------------------
        def make_cross_attn():
            return PreNorm(
                latent_dim,
                MHSAtt(
                    latent_dim,
                    context_dim=latent_dim,
                    heads=cross_head,
                    dropout=attn_dropout,
                    temperature=temperature,
                    use_leaky_out=use_leaky_out,
                    use_gate=use_gate
                ),
                context_dim=latent_dim
            )

        def make_latent_attn():
            return PreNorm(
                latent_dim,
                MHSAtt(
                    latent_dim,
                    heads=latent_heads,
                    dropout=attn_dropout,
                    temperature=temperature,
                    use_leaky_out=use_leaky_out,
                    use_gate=use_gate
                )
            )

        def make_latent_ff():
            return PreNorm(
                latent_dim,
                FFN(latent_dim, dropout=ff_dropout, snn=snn)
            )

        def make_cross_ff():
            return PreNorm(
                latent_dim,
                FFN(latent_dim, dropout=ff_dropout, snn=snn)
            )

        # ----------------------
        # Build Layers
        # ----------------------
        self.layers = nn.ModuleList([])

        for i in range(depth):
            # Weight sharing
            shared = (i > 0 and weight_tie_layers)

            cross_blocks = nn.ModuleList()
            for m in range(n_modalities):
                cross_blocks.append(make_cross_attn())
                cross_blocks.append(make_cross_ff())

            # latent self-attn blocks
            latent_blocks = nn.ModuleList()
            for _ in range(self_per_cross_attn):
                latent_blocks.append(make_latent_attn())
                latent_blocks.append(make_latent_ff())

            self.layers.append(nn.ModuleList([cross_blocks, latent_blocks]))

        # ----------------------
        # Final prediction
        # ----------------------
        self.to_logits = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, out_dims)
        )

        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ----------------------------------------
    # Forward
    # ----------------------------------------
    def forward(self, embeddings: List[Optional[torch.Tensor]]):
        # proj all modalities
        contexts = []
        for t in embeddings:
            ctx = self.shared_proj(t)
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(1)
            contexts.append(ctx)

        B = embeddings[0].shape[0]

        # replicate latent tokens
        x = self.latents.unsqueeze(0).repeat(B, 1, 1)

        # layer-by-layer fusion
        for cross_blocks, latent_blocks in self.layers:

            # cross attention部分
            k = 0
            for m in range(self.n_modalities):
                cross_attn = cross_blocks[k]
                cross_ff = cross_blocks[k + 1]
                k += 2

                x = x + cross_attn(x, context=contexts[m])
                x = x + cross_ff(x)

            # latent self-attn部分
            if len(latent_blocks) > 0:
                x = x + latent_blocks[0](x)
                x = x + latent_blocks[1](x)

        # latent pooling
        x_out = x.mean(dim=1) # [B, latent_dim]
        # return self.to_logits(x_out)
        return x_out

    def get_attention_weights(self):
        ws = []
        for m in self.modules():
            if isinstance(m, MHSAtt) and m.attn_weights is not None:
                ws.append(m.attn_weights)
        return ws


#########################################
# ModalityCompress
#########################################

class ModalityCompress_module(nn.Module):
    def __init__(self,
                 z_dim=256,
                 beta=1e-2,
                 sample_num=50,
                 topk=256,
                 num_classes=4,
                 seed=1):
        super().__init__()

        self.beta = beta
        self.sample_num = sample_num
        self.topk = topk
        self.num_classes = num_classes
        self.z_dim = z_dim

        # ============ Decoder (z -> y) ============ #
        self.decoder_logits = nn.Linear(z_dim, num_classes)

        # ============ Proxies (2*C classes) ============ #
        self.z_proxy = nn.Parameter(
            torch.zeros(num_classes * 2, sample_num, z_dim),
            requires_grad=True
        )
        torch.nn.init.xavier_uniform_(self.z_proxy)

        # proxy index mapping: censor,label → proxy id
        self.proxies_dict = {
            f"{c},{y}": c * num_classes + y
            for c in range(2) for y in range(num_classes)
        }

        self.tau = 0.1  # temperature
        self.lambda_ib = 1.0  # I(Z,X)

        # SurvLoss
        self.loss_surv = NLLSurvLoss(alpha=0.5)

    def get_loss_proxy(self, x, loss):
        censor = torch.empty([self.num_classes * 2]).cuda()
        for i in range(self.num_classes):
            censor[i] = 0
            censor[i + self.num_classes] = 1
        y = torch.arange(0, self.num_classes).repeat(2).cuda()
        loss_proxy = loss(h=x, y=y, t=None, c=censor)

        return loss_proxy

    def forward(self, x, y=None, c=None, return_att=False):
        """
        input:
            x: [B, N, x_dim]
            y: label
            c: censor
        """

        B, N, _ = x.shape
        z_norm = F.normalize(x, dim=-1)     # [B, N, z_dim]
        
        proxy_logits = torch.mean(self.decoder_logits(self.z_proxy), dim=1)  # [2C, C]

        loss_I_ZY = self.get_loss_proxy(proxy_logits, self.loss_surv)

        proxy_state = F.normalize(self.z_proxy.mean(dim=1), dim=-1)         # [2C, d]
        z_proxy_norm = torch.unsqueeze(proxy_state, dim=0)  # [1, 2C, d]
        att = torch.matmul(z_norm, torch.transpose(z_proxy_norm, 1, 2))  # [B, N, 2C]

        if y is not None and c is not None:
            proxy_indices = torch.tensor([
                self.proxies_dict[f"{int(ci)},{int(yi)}"]
                for ci, yi in zip(c, y)
            ]).long().cuda()  # shape [B]
            pos_idx = int(proxy_indices.item())
            
            # att: [B, N, 2C]
            att_scaled = att / self.tau

            # p(z | x): soft assignment
            p_z_given_x = F.softmax(att_scaled, dim=-1)  # [B, N, 2C]

            # entropy H(Z | X)
            entropy = -torch.sum(
                p_z_given_x * torch.log(p_z_given_x + 1e-8),
                dim=-1
            )  # [B, N]

            loss_I_ZX = entropy.mean() * self.lambda_ib

            # marginal p(z)
            p_z = p_z_given_x.mean(dim=(0, 1))  # [2C]

            # uniform prior or learned prior
            prior = torch.ones_like(p_z) / p_z.numel()

            loss_kl = F.kl_div(
                torch.log(p_z + 1e-8),
                prior,
                reduction='batchmean'
            )
            # loss_kl = torch.tensor(0.).cuda()

            # loss_kl=0.0
            loss_I_ZX = loss_I_ZX + 0.1 * loss_kl

        else:
            att_unbind_proxy = torch.cat(torch.unbind(att, dim=1), dim=1)  # [B, M, P]-->M*[B, 1, P]-->[B, M*P]
            att_topk_proxy_logits, att_topk_proxy_idx = torch.topk(att_unbind_proxy, self.topk, dim=1)  # [B, topk]
            att_topk_proxy_idx = att_topk_proxy_idx % (self.num_classes * 2)
            positive_proxy_idx, _ = torch.mode(att_topk_proxy_idx, dim=1)

            self.pos_idx = positive_proxy_idx
            self.pos_logits = att_topk_proxy_logits.mean()
            pos_idx = int(positive_proxy_idx.item())
            loss_I_ZX = None

        att_flat = torch.cat(torch.unbind(att, dim=2), dim=1)  # [B, N*2C]
        _, topk_idx = torch.topk(att_flat, self.topk, dim=1)
        topk_idx = topk_idx % N # [B, topk]

        z_topk = torch.gather(
            x, 1,
            topk_idx.unsqueeze(-1).repeat(1, 1, self.z_dim) # [B, topk, z_dim]
        )

        proxy_pos = self.z_proxy[pos_idx].unsqueeze(0) # [1, sample, z_dim]
        # if proxy_pos.dim() == 1:
        #     proxy_pos = proxy_pos.unsqueeze(0)
        if return_att:
            return loss_I_ZY, loss_I_ZX, z_topk, proxy_pos, att
        else:
            return loss_I_ZY, loss_I_ZX, z_topk, proxy_pos


class MCFD(nn.Module):
    def __init__(
            self,
            wsi_input_dim,
            omic_input_dim,
            model_size_geno: str = 'small',
            proj_dim=256,
            num_classes=4,
            proxy_num=50,
            topk_wsi=256,
            depth=2,
            latent_num=17,
            cross_head=4,
            latent_heads=4,
            use_mlp=False,
            dropout=0.1,
            alpha=0.1,
            beta=0.01,
            seed=1,
    ):
        super().__init__()

        # ---> general props
        self.num_classes = num_classes
        self.sample_num = proxy_num
        self.alpha = alpha
        self.beta = beta
        self.wsi_embedding_dim = wsi_input_dim
        self.proj_dim = proj_dim

        # ---> wsi encoder
        self.wsi_encoder = nn.Sequential(
            nn.Linear(self.wsi_embedding_dim, self.proj_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.proj_dim * 2, self.proj_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.proj_dim * 2, self.proj_dim),
        )

        # ---> gene encoder
        self.size_dict_geno = {'small': [1024, 256], 'big': [1024, 1024, 1024, 256]}
        hidden = self.size_dict_geno[model_size_geno]
        if use_mlp:
            Block = MLP_Block
        else:
            Block = SNN_Block
        fc_omic = [Block(dim1=omic_input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
        self.fc_omic = nn.Sequential(*fc_omic)

        # ---> FR
        self.topk_wsi = topk_wsi
        self.FR_wsi = ModalityCompress_module(z_dim=self.proj_dim, num_classes=self.num_classes,
                                topk=self.topk_wsi, sample_num=self.sample_num, seed=seed)
        # ---> TRIE
        self.get_latent = FeatureDistillation_module(
            embed_dim=self.proj_dim,
            out_dims=self.num_classes,
            n_modalities=2,
            depth=depth,
            latent_num = latent_num,  # default=17
            latent_dim = self.proj_dim,  # default=128
            cross_head = cross_head,  # default=1
            latent_heads = latent_heads,  # default=8
        )

        # ---> classifier
        self.to_logits = nn.Sequential(
            nn.Linear(self.proj_dim, self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, self.num_classes),
        )

    def forward(self, **kwargs):
        wsi = kwargs['x_path']
        x_geno = kwargs['x_omic']
        y = kwargs["label"]
        c = kwargs["censor"]
        training = kwargs["training"] # True or False
        return_att = kwargs.get("return_att", False)
        h_wsi = self.wsi_encoder(wsi).unsqueeze(dim=0) #  [1, n_p, d]
        h_omic = self.fc_omic(x_geno.unsqueeze(dim=0)).unsqueeze(dim=0) #  [1, 1, d]

        loss_wsi_IB = 0.0

        if return_att:
            loss_zy_wsi, loss_zx_wsi, f_topk_wsi, proto_wsi, att = self.FR_wsi(h_wsi, y=y, c=c, return_att=return_att)
        else:
            loss_zy_wsi, loss_zx_wsi, f_topk_wsi, proto_wsi = self.FR_wsi(h_wsi, y=y, c=c, return_att=return_att)

        z_latent = self.get_latent([h_omic, f_topk_wsi])
        logits = self.to_logits(z_latent)

        if training:
            loss_wsi_IB = self.alpha * loss_zy_wsi + self.beta * loss_zx_wsi

        Y_hat = torch.topk(logits, 1, dim=1)[1]  # [1, 1]
        hazards = torch.sigmoid(logits)  # [1, num_classer]
        S = torch.cumprod(1 - hazards, dim=1)  # [1, num_classer]
        if return_att:
            return hazards, S, Y_hat, logits, loss_wsi_IB, att
        else:
            return hazards, S, Y_hat, logits, loss_wsi_IB



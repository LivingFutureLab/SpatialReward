import torch
from torch import nn
import torch.nn.init as init
from typing import Optional
from transformers import AutoModel, Dinov2Model
from transformers.models.dinov2.modeling_dinov2 import Dinov2Embeddings
from transformers.models.dinov2.configuration_dinov2 import Dinov2Config
from contextlib import nullcontext

from ..constants import DINO_SMALL, DINO_BASE, DINO_LARGE, DINO_GIANT


def get_activation(activation):
    activations = {
        "gelu": nn.GELU,
        "rrelu": lambda: nn.RReLU(inplace=True),
        "selu": lambda: nn.SELU(inplace=True),
        "silu": lambda: nn.SiLU(inplace=True),
        "hardswish": lambda: nn.Hardswish(inplace=True),
        "leakyrelu": lambda: nn.LeakyReLU(inplace=True),
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
    }
    factory = activations.get(activation.lower())
    if factory:
        return factory()
    return nn.ReLU(inplace=True)


class MLP_dim(nn.Module):
    def __init__(self, in_dim=512, out_dim=1024, bias=True, activation="relu"):
        super().__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Linear(in_dim, int(out_dim), bias=bias),
            nn.BatchNorm1d(int(out_dim)),
            self.act,
        )
        self.net2 = nn.Sequential(
            nn.Linear(int(out_dim), out_dim, bias=bias),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x):
        return self.net2(self.net1(x))


class FLIP_Dinov2Embeddings(Dinov2Embeddings):
    """CLS token, mask token, position and patch embeddings with FLIP masking support."""

    def __init__(self, config: Dinov2Config) -> None:
        super().__init__(config)

    def forward(
        self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)

        if bool_masked_pos is not None:
            B, S, D = embeddings.shape
            batch_indices = torch.arange(B).unsqueeze(1)
            embeddings = embeddings[batch_indices, bool_masked_pos]

        embeddings = self.dropout(embeddings)
        return embeddings


class FLIP_DINOv2(Dinov2Model):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = FLIP_Dinov2Embeddings(config)


class DINOv2_MLP(nn.Module):
    """DINOv2 backbone with MLP head for orientation prediction."""

    def __init__(
        self,
        dino_mode: str,
        in_dim: int,
        out_dim: int,
        evaluate: bool,
        mask_dino: bool,
        frozen_back: bool,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()

        dino_models = {
            "base": DINO_BASE,
            "large": DINO_LARGE,
            "small": DINO_SMALL,
            "giant": DINO_GIANT,
        }
        model_id = dino_models.get(dino_mode, DINO_LARGE)
        self.dinov2 = FLIP_DINOv2.from_pretrained(model_id, cache_dir=cache_dir)

        self.down_sampler = MLP_dim(in_dim=in_dim, out_dim=out_dim)
        self.random_mask = False
        if not evaluate:
            self.init_weights(self.down_sampler)
            self.random_mask = mask_dino
        if frozen_back:
            self.forward_mode = torch.no_grad()
        else:
            self.forward_mode = nullcontext()

    def forward(self, img_inputs):
        device = self.get_device()

        with self.forward_mode:
            if self.random_mask:
                B = len(img_inputs["pixel_values"])
                S = 256
                indices = []
                for i in range(B):
                    tmp = torch.randperm(S)[: S // 2]
                    tmp = tmp.sort().values + 1
                    indices.append(tmp)
                indices = torch.stack(indices, dim=0)
                indices = torch.cat(
                    [torch.zeros(B, 1, dtype=torch.long, device="cpu"), indices], dim=1
                )
                img_inputs["bool_masked_pos"] = indices.to(device)

            dino_outputs = self.dinov2(**img_inputs)
            dino_seq = dino_outputs.last_hidden_state
            dino_seq = dino_seq[:, 0, :]

        down_sample_out = self.down_sampler(dino_seq)
        return down_sample_out

    def get_device(self):
        return next(self.parameters()).device

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)

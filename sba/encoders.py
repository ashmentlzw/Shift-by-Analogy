"""SBA visual/text encoders: frozen CLIP plus scaled parallel FFN adapters."""
from functools import partial
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class DirectionBank(nn.Module):
    def __init__(self, dimension, rank):
        super().__init__()
        self.down = nn.Linear(dimension, rank)
        self.up = nn.Linear(rank, dimension, bias=False)
        nn.init.normal_(self.down.weight, std=1 / math.sqrt(dimension))
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)

    def forward(self, unit):
        return self.up(F.gelu(self.down(unit)))


class ParallelOffset(nn.Module):
    def __init__(self, dimension, rank):
        super().__init__()
        self.dimension, self.rank = dimension, rank
        self.shared = DirectionBank(dimension, rank)

    def forward(self, h):
        with torch.autocast(device_type=h.device.type, enabled=False):
            full = h if h.dtype == torch.float64 else h.float()
            flat = full.reshape(-1, self.dimension)
            norm = flat.norm(dim=-1, keepdim=True)
            unit = flat / norm.clamp_min(torch.finfo(flat.dtype).tiny) * math.sqrt(self.dimension)
            result = (norm * self.shared(unit) / math.sqrt(self.rank)).reshape_as(full)
        return result.to(h.dtype)


def adapted_block(block, adapter, h):
    attended = h + block.attention(block.ln_1(h))
    normalized = block.ln_2(attended)
    return attended + block.mlp(normalized) + .1 * adapter(normalized)


class TaskEncoder(nn.Module):
    def __init__(self, clip, rank=64):
        super().__init__()
        self.clip = clip.eval().requires_grad_(False)
        vd, td = clip.visual.conv1.out_channels, clip.token_embedding.weight.shape[1]
        nv, nt = len(clip.visual.transformer.resblocks), len(clip.transformer.resblocks)
        # Preserve the original experiment's RNG draw order. The first banks are
        # replaced immediately; they are not part of the trained model/state.
        self.visual_adapters = nn.ModuleList([DirectionBank(vd, rank) for _ in range(nv)])
        self.text_adapters = nn.ModuleList([DirectionBank(td, rank) for _ in range(nt)])
        for modules, dimension in ((self.visual_adapters, vd), (self.text_adapters, td)):
            for i in range(len(modules)):
                modules[i] = ParallelOffset(dimension, rank)

    def train(self, mode=True):
        super().train(mode)
        self.clip.eval()
        return self

    def adaptation_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if not k.startswith('clip.')}

    def load_adaptation_state(self, state):
        expected = {k for k in self.state_dict() if not k.startswith('clip.')}
        if set(state) != expected or any(not torch.isfinite(v).all() for v in state.values()):
            raise ValueError('Incomplete, unexpected or nonfinite adaptation state')
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(not k.startswith('clip.') for k in result.missing_keys):
            raise ValueError('Adaptation state mismatch')

    def _visual_step(self, index, h):
        return adapted_block(self.clip.visual.transformer.resblocks[index], self.visual_adapters[index], h)

    def encode_image(self, pixels, checkpoint_blocks=True):
        v = self.clip.visual
        x = v.conv1(pixels.to(self.clip.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = v.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device, dtype=x.dtype)
        x = v.ln_pre(torch.cat([cls, x], dim=1) + v.positional_embedding.to(x.dtype)).permute(1, 0, 2)
        for i in range(len(self.visual_adapters)):
            x = (checkpoint(partial(self._visual_step, i), x, use_reentrant=False)
                 if checkpoint_blocks and torch.is_grad_enabled() else self._visual_step(i, x))
        x = v.ln_post(x.permute(1, 0, 2))
        return F.normalize((x @ v.proj)[:, 0].float(), dim=-1)

    def _text_chunk(self, ids):
        x = self.clip.token_embedding(ids).to(self.clip.dtype) + self.clip.positional_embedding.to(self.clip.dtype)
        x = x.permute(1, 0, 2)
        for i, block in enumerate(self.clip.transformer.resblocks):
            x = adapted_block(block, self.text_adapters[i], x)
        x = self.clip.ln_final(x.permute(1, 0, 2)).to(self.clip.dtype)
        features = x[torch.arange(len(ids), device=ids.device), ids.argmax(-1)] @ self.clip.text_projection
        return F.normalize(features.float(), dim=-1)

"""SBA's joint task encoder and analogy/family proxy scorer."""
import hashlib
import random
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .encoders import TaskEncoder
from .proxy import JointProxy

PUBLIC_CLIP_SHA256 = 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_clip(path, device):
    import clip
    if sha256(path) != PUBLIC_CLIP_SHA256:
        raise ValueError('Expected the public OpenAI CLIP ViT-L/14 checkpoint; see RUNNING.md')
    model, _ = clip.load(str(path), device='cpu', jit=False)
    model = model.float().eval().requires_grad_(False)
    model.context_length = 32
    model.positional_embedding = torch.nn.Parameter(model.positional_embedding[:32].detach(), requires_grad=False)
    for block in model.transformer.resblocks:
        block.attn_mask = block.attn_mask[:32, :32]
    return model.to(device)


def tokenize_pairs(pairs):
    import clip
    prompts = ['a photo of ' + a.replace('.', ' ').lower() + ' ' + o.replace('.', ' ').lower()
               for a, o in pairs]
    return clip.tokenize(prompts, context_length=32, truncate=False)


def tokenize_primitives(attrs, objs):
    import clip
    return clip.tokenize(['a photo of ' + x.replace('.', ' ').replace('_', ' ').lower()
                          for x in attrs + objs], context_length=32, truncate=False)


class SBA(TaskEncoder):
    def __init__(self, clip_model, asset, seed=42, gain=0.1, image_checkpoint=False):
        if gain <= 0:
            raise ValueError('Gain must be positive')
        # Keep historical constructor order, including overwritten modules, so seed42
        # produces exactly the released experiment's adapter initialization.
        seed_all(seed)
        super().__init__(clip_model, rank=64)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.proxy_network = JointProxy(asset, gain=gain)
        self.image_checkpoint = image_checkpoint

    def adapter_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def encode_image(self, pixels, checkpoint_blocks=None):
        use_checkpoint = self.image_checkpoint if checkpoint_blocks is None else checkpoint_blocks
        return self.proxy_network.visual(super().encode_image(pixels, use_checkpoint))

    def encode_candidates(self, tokens, ids, checkpoint_chunks=True, chunk_size=64):
        outputs = []
        for tok, index in zip(tokens.split(chunk_size), ids.split(chunk_size)):
            def encode(t, i):
                return self.proxy_network.text(self._text_chunk(t), i)
            outputs.append(checkpoint(encode, tok, index, use_reentrant=False)
                           if checkpoint_chunks and torch.is_grad_enabled() else encode(tok, index))
        return torch.cat(outputs)

    def loss(self, pixels, tokens, ids, labels, alpha):
        image = self.encode_image(pixels)
        text = self.encode_candidates(tokens, ids)
        scores = self.proxy_network.scores(image, text, self.clip.logit_scale.exp())
        pairs = self.proxy_network.pair_ids[ids][labels]
        main = F.cross_entropy(scores['main'], labels)
        fused = F.cross_entropy(scores['fused'], labels)
        auxiliary = (F.cross_entropy(scores['visual'], labels)
                     + F.cross_entropy(scores['attribute'], pairs[:, 0])
                     + F.cross_entropy(scores['object'], pairs[:, 1]))
        return .5 * (main + fused) + .1 * alpha * auxiliary

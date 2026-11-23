"""Build a support bank from public CLIP and TRAIN images only."""
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .data import CompositionData, Images
from .model import load_clip, tokenize_pairs, tokenize_primitives, PUBLIC_CLIP_SHA256
from .support import build_bank
from .utils import configure, save_checkpoint


@torch.no_grad()
def run(args):
    configure(args.threads)
    if Path(args.output).exists():
        raise FileExistsError('Support bank already exists; choose a new output')
    data = CompositionData(args.data_root)
    clip = load_clip(args.clip, args.device)
    train = data.train_pairs
    tokens = tokenize_pairs(train)
    text = torch.cat([F.normalize(clip.encode_text(t.to(args.device)), dim=-1).cpu()
                      for t in tokens.split(128)])
    loader = DataLoader(Images(data, 'train', train), batch_size=args.batch_size,
                        num_workers=args.workers, shuffle=False)
    images, labels = [], []
    for step, (pixels, y) in enumerate(loader, 1):
        images.append(F.normalize(clip.encode_image(pixels.to(args.device)), dim=-1).cpu())
        labels.append(y)
        if step == 1 or step % 50 == 0 or step == len(loader):
            print(f'TRAIN support extraction {step}/{len(loader)}', flush=True)
    images, labels = torch.cat(images), torch.cat(labels)
    counts = torch.bincount(labels, minlength=len(train))
    if (counts == 0).any():
        raise ValueError('A training composition has no training images')
    means = torch.zeros_like(text).index_add_(0, labels, images) / counts[:, None]
    train_set = set(train)
    pairs = train + [p for p in data.open_pairs if p not in train_set]
    bank = build_bank(train, pairs, text, means)
    primitive_tokens = tokenize_primitives(data.attrs, data.objs)
    primitive = torch.cat([F.normalize(clip.encode_text(t.to(args.device)), dim=-1).cpu()
                           for t in primitive_tokens.split(64)])
    ai, oi = {a: i for i, a in enumerate(data.attrs)}, {o: i for i, o in enumerate(data.objs)}
    save_checkpoint(args.output, dict(format_version=1, bank=bank, attrs=data.attrs, objs=data.objs,
        primitive_text=primitive, pairs=pairs, train_pairs=train,
        pair_ids=torch.tensor([(ai[a], oi[o]) for a, o in pairs]),
        public_clip_sha256=PUBLIC_CLIP_SHA256, train_images=len(images),
        val_images_used=False, test_images_used=False))
    print(f'Saved {args.output}: {len(train)} support pairs, {len(pairs)} candidate pairs', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--clip', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--threads', type=int, default=4)
    run(p.parse_args())


if __name__ == '__main__':
    main()

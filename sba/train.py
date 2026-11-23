"""Train SBA from public CLIP, selecting checkpoints by closed VAL AUC."""
import argparse
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data import CompositionData, Images
from .evaluate import predict, measure
from .model import SBA, load_clip, tokenize_pairs, seed_all, sha256
from .optim import sam_step
from .utils import configure, save_checkpoint, save_json


def learning_rate(epoch, step, steps):
    position = (epoch - 1) * steps + step - 1
    if epoch == 1:
        return 1e-5 + (1e-4 - 1e-5) * position / max(steps - 1, 1)
    progress = (position - steps) / max(12 * steps - steps - 1, 1)
    return 1e-5 + (1e-4 - 1e-5) * .5 * (1 + math.cos(math.pi * progress))


def run(args):
    configure(args.threads)
    if not 1 <= args.epochs <= 12:
        raise ValueError('The formal schedule is 12 epochs; --epochs can stop it earlier')
    out = Path(args.output)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError('Use a new output directory or --resume last.pt')
    out.mkdir(parents=True, exist_ok=True)
    data = CompositionData(args.data_root)
    asset = torch.load(args.bank, weights_only=True, map_location='cpu')
    if data.attrs != asset['attrs'] or data.objs != asset['objs'] or data.train_pairs != list(map(tuple, asset['train_pairs'])):
        raise ValueError('Dataset and support bank disagree')
    bank_hash = sha256(args.bank)
    config = dict(seed=args.seed, gain=args.gain, batch_size=args.batch_size,
                  image_checkpoint=args.image_checkpoint, schedule_epochs=12)
    model = SBA(load_clip(args.clip, 'cpu'), asset, seed=args.seed,
                gain=args.gain, image_checkpoint=args.image_checkpoint).to(args.device)
    optimizer = torch.optim.AdamW(model.adapter_parameters(), lr=1e-5, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    history, first, best_auc = [], 1, -1.
    if args.resume:
        state = torch.load(args.resume, weights_only=True, map_location='cpu')
        if state['bank_sha256'] != bank_hash or state['config'] != config:
            raise ValueError('Resume configuration/support bank differs from the checkpoint')
        model.load_adaptation_state(state['adapters'])
        optimizer.load_state_dict(state['optimizer'])
        history, first, best_auc = state['history'], state['epoch'] + 1, state['best_auc']
        if Path(args.resume).resolve().parent != out.resolve():
            raise ValueError('Resume into the original run directory (retains best.pt)')
        if not (out / 'best.pt').is_file():
            raise FileNotFoundError('Original best.pt must be retained for resume')
    pairs = data.train_pairs
    lookup = {tuple(p): i for i, p in enumerate(asset['pairs'])}
    ids = torch.tensor([lookup[p] for p in pairs], device=args.device)
    tokens = tokenize_pairs(pairs).to(args.device)
    save_json(out / 'config.json', config)
    for epoch in range(first, args.epochs + 1):
        seed_all(args.seed + epoch)
        reader = Images(data, 'train', pairs, augment=True, epoch=epoch, seed=args.seed)
        order = torch.randperm(len(reader), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        loader = DataLoader(reader, batch_size=args.batch_size, sampler=order, num_workers=args.workers,
                            pin_memory=args.device.startswith('cuda'), drop_last=False,
                            generator=torch.Generator().manual_seed(args.seed + epoch))
        model.train()
        loss_sum = 0.
        for step, (pixels, labels) in enumerate(loader, 1):
            pixels, labels = pixels.to(args.device), labels.to(args.device)
            lr = learning_rate(epoch, step, len(loader))
            for group in optimizer.param_groups:
                group['lr'] = lr
            alpha = (step - 1) / max(len(loader) - 1, 1) if epoch == 1 else 1.
            result = sam_step(model.adapter_parameters(), optimizer,
                              lambda: model.loss(pixels, tokens, ids, labels, alpha), rho=.05, max_grad_norm=5.)
            loss_sum += result['ce'] * len(labels)
            if step == 1 or step % 25 == 0 or step == len(loader):
                print(f'epoch {epoch}/12 step {step}/{len(loader)} loss={result["ce"]:.5f} lr={lr:.7f}', flush=True)
        scores, info = predict(model, asset, data, 'val', 'closed', device=args.device,
                               batch_size=args.batch_size, workers=args.workers)
        metric = measure(scores, info)
        del scores
        row = dict(epoch=epoch, train_loss=loss_sum / len(reader), validation=metric)
        history.append(row)
        improved = metric['AUC'] > best_auc  # First epoch wins exact ties.
        best_auc = max(best_auc, metric['AUC'])
        state = dict(format_version=1, epoch=epoch, adapters=model.adaptation_state(),
                     optimizer=optimizer.state_dict(), config=config, bank_sha256=bank_hash,
                     history=history, best_auc=best_auc, selection='closed VAL AUC')
        if improved:
            save_checkpoint(out / 'best.pt', state)
        save_checkpoint(out / 'last.pt', state)
        save_json(out / 'history.json', history)
        print(f'epoch {epoch}: VAL AUC={100 * metric["AUC"]:.4f}; best={100 * best_auc:.4f}', flush=True)
    print(f'Checkpoints: {out / "best.pt"} and {out / "last.pt"}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'clip', 'bank', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--resume', help='last.pt from the same run; resumes at the next complete epoch')
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gain', type=float, default=.1)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--image-checkpoint', action='store_true', help='Trade compute for lower activation memory')
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--threads', type=int, default=4)
    run(p.parse_args())


if __name__ == '__main__':
    main()

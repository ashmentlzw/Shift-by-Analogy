"""Closed/open CZSL evaluation; optional VAL-selected feasibility filtering."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import CompositionData, Images
from .metrics import Curve
from .model import SBA, load_clip, tokenize_pairs, sha256
from .utils import configure, save_json


@torch.no_grad()
def predict(model, asset, data, phase, setting, *, device, batch_size=64, workers=4, score_path=None):
    model.eval()
    pairs = data.candidates(setting)
    lookup = {tuple(p): i for i, p in enumerate(asset['pairs'])}
    ids = torch.tensor([lookup[p] for p in pairs])
    tokens = tokenize_pairs(pairs)
    text = []
    for tok, index in zip(tokens.split(64), ids.split(64)):
        text.append(model.encode_candidates(tok.to(device), index.to(device), False).cpu())
    text = torch.cat(text)
    loader = DataLoader(Images(data, phase, pairs), batch_size=batch_size, num_workers=workers, shuffle=False)
    n = len(loader.dataset)
    array = (np.lib.format.open_memmap(str(score_path), mode='w+', dtype='float32', shape=(n, len(pairs)))
             if score_path else np.empty((n, len(pairs)), dtype=np.float32))
    truth, offset = [], 0
    for step, (pixels, y) in enumerate(loader, 1):
        image = model.encode_image(pixels.to(device), checkpoint_blocks=False)
        for start in range(0, len(pairs), 8192):
            value = model.clip.logit_scale.exp() * (image @ text[start:start + 8192].to(device).T)
            array[offset:offset + len(y), start:start + value.shape[1]] = value.cpu().numpy()
        truth.append(y); offset += len(y)
        if step == 1 or step % 50 == 0 or step == len(loader):
            print(f'{phase} {setting}: {step}/{len(loader)} batches', flush=True)
    if hasattr(array, 'flush'):
        array.flush()
    train_set = set(data.train_pairs)
    info = dict(truth=torch.cat(truth), train_rows=torch.tensor([i for i, p in enumerate(pairs) if p in train_set]),
                mask=data.mask(pairs, phase, setting))
    return torch.from_numpy(array), info


def measure(scores, info):
    return Curve(scores, info['truth'], info['train_rows'], info['mask']).evaluate()


def filtered_scores(scores, feasibility, seen, threshold):
    # Reproduce the published zero-score multiplicative protocol, including
    # addition of seen_mask. This deliberately is not -inf candidate removal.
    return scores * ((feasibility >= threshold).float() + seen.float())


def run(args):
    configure(args.threads)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if (out / 'report.json').exists():
        raise FileExistsError('Evaluation output already completed; use a new directory')
    asset = torch.load(args.bank, map_location='cpu', weights_only=True)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if state['bank_sha256'] != sha256(args.bank):
        raise ValueError('Checkpoint belongs to another support bank')
    data = CompositionData(args.data_root)
    if data.attrs != asset['attrs'] or data.objs != asset['objs'] or data.train_pairs != list(map(tuple, asset['train_pairs'])):
        raise ValueError('Dataset vocabulary or training split differs from support bank')
    gain = state['config']['gain'] if args.gain is None else args.gain
    model = SBA(load_clip(args.clip, 'cpu'), asset, seed=state['config']['seed'], gain=gain).to(args.device)
    model.load_adaptation_state(state['adapters'])
    kwargs = dict(device=args.device, batch_size=args.batch_size, workers=args.workers)
    threshold = None; trials = []; feasibility = seen = None
    if args.feasibility:
        if args.setting != 'open':
            raise ValueError('Feasibility filtering requires --setting open')
        obj = torch.load(args.feasibility, map_location='cpu', weights_only=True)
        feasibility = obj['feasibility'].float()
        if feasibility.shape != (len(data.open_pairs),) or not torch.isfinite(feasibility).all():
            raise ValueError('Feasibility must match the full sorted attribute × object universe')
        train_set = set(data.train_pairs)
        seen = torch.tensor([p in train_set for p in data.open_pairs])
        if not torch.equal(feasibility[seen], torch.zeros(int(seen.sum()))):
            raise ValueError('Expected zero feasibility values for seen pairs (Troika convention)')
        val, vi = predict(model, asset, data, 'val', 'open', score_path=out / 'val_scores.npy', **kwargs)
        for index, t in enumerate(torch.linspace(feasibility[~seen].min(), feasibility[~seen].max(), 50).tolist()):
            metric = measure(filtered_scores(val, feasibility, seen, t), vi)
            trials.append(dict(index=index, threshold=t, metrics=metric))
            save_json(out / 'threshold_trials.json', trials)
            print(f'VAL threshold {index + 1}/50 AUC={metric["AUC"] * 100:.4f}', flush=True)
        selected = max(trials, key=lambda r: (r['metrics']['AUC'], -r['index']))
        threshold = selected['threshold']
        save_json(out / 'selected_threshold.json', selected)
        del val
    scores, info = predict(model, asset, data, args.phase, args.setting,
                           score_path=out / f'{args.phase}_scores.npy', **kwargs)
    if threshold is not None:
        scores = filtered_scores(scores, feasibility, seen, threshold)
    result = measure(scores, info)
    report = dict(phase=args.phase, setting=args.setting, feasibility_filter=bool(args.feasibility),
        checkpoint_sha256=sha256(args.checkpoint), bank_sha256=state['bank_sha256'],
        epoch=state['epoch'], gain=gain, gain_overridden=args.gain is not None,
        threshold=threshold, threshold_selection='open VAL AUC' if args.feasibility else None,
        feasibility_sha256=sha256(args.feasibility) if args.feasibility else None,
        metrics_fraction=result, metrics_percent={k: 100 * result[k] for k in ('best_seen', 'best_unseen', 'best_hm', 'AUC')})
    save_json(out / 'report.json', report)
    print(report['metrics_percent'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'clip', 'bank', 'checkpoint', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--setting', choices=['closed', 'open'], default='closed')
    p.add_argument('--phase', choices=['val', 'test'], default='test')
    p.add_argument('--feasibility', help='Troika-format .pt; select threshold on VAL, then freeze for TEST')
    p.add_argument('--gain', type=float, help='Optional explicit inference gain override; recorded in report')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--threads', type=int, default=4)
    run(p.parse_args())


if __name__ == '__main__':
    main()

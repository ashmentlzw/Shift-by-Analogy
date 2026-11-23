"""Small CPU checks for support exclusion, optimization, metrics and run recovery."""
import argparse
from itertools import product
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from sba.support import build_bank
from sba.metrics import Curve
from sba.optim import sam_step
from sba.data import CompositionData
from sba import prepare, train, evaluate


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(3)

    def test_target_exclusion_and_three_corner_estimate(self):
        pairs = list(product(['blue', 'red'], ['boat', 'car']))
        text, means = torch.randn(4, 8), torch.randn(4, 8)
        bank = build_bank(pairs, pairs, text, means)
        residual = means - text
        for target, (a, o) in enumerate(pairs):
            supports = bank['local_ids'][target, bank['local_mask'][target]].tolist()
            self.assertNotIn(target, supports)
            b, p = next(b for b in ['blue', 'red'] if b != a), next(p for p in ['boat', 'car'] if p != o)
            expected = residual[pairs.index((a, p))] - residual[pairs.index((b, p))] + residual[pairs.index((b, o))]
            torch.testing.assert_close(bank['cross'][target], expected, rtol=0, atol=0)
            altered = means.clone(); altered[target] += 1000
            other = build_bank(pairs, pairs, text, altered)
            torch.testing.assert_close(other['cross'][target], bank['cross'][target], rtol=0, atol=0)

    def test_sam_restores_parameters_on_failed_second_closure(self):
        p = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([p], lr=.1)
        calls = []
        def closure():
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError('test interruption')
            return p.square().sum()
        with self.assertRaises(RuntimeError):
            sam_step([p], optimizer, closure)
        torch.testing.assert_close(p, torch.ones(1), rtol=0, atol=0)
        self.assertFalse(optimizer.state)

    def test_curve_matches_full_topk_with_float32_ties(self):
        for rounded in (False, True):
            scores = torch.randn(50, 9)
            if rounded:
                scores = scores.round(); scores[:, 5] = 0
            truth = torch.arange(50) % 8
            rows = torch.tensor([0, 1, 2]); mask = torch.tensor([True] * 8 + [False])
            seen_columns = torch.zeros(9, dtype=torch.bool); seen_columns[rows] = True
            seen = seen_columns[truth]
            def hits(bias):
                z = scores.clone(); z[:, ~seen_columns] += bias; z[:, ~mask] = -1e10
                return z.topk(1, dim=1).indices[:, 0] == truth
            last = hits(1000.)
            difference = (scores[:, rows].max(1).values - scores[torch.arange(50), truth] - 1e-4)[~seen & last].sort().values
            biases = torch.cat([difference[::max(len(difference)//20, 1)], torch.tensor([1000.])])
            sa, ua = [], []
            for bias in biases:
                correct = hits(bias)
                sa.append(float(correct[seen].float().mean())); ua.append(float(correct[~seen].float().mean()))
            result = Curve(scores, truth, rows, mask).evaluate()
            self.assertEqual(result['AUC'], float(np.trapz(sa, ua)))
            self.assertEqual(result['best_seen'], max(sa))
            self.assertEqual(result['best_unseen'], max(ua))

    def test_prepare_train_resume_and_evaluate(self):
        # Tiny CLIP exercises the real entrypoints without downloading weights.
        from clip.model import CLIP
        def tiny_clip(*args):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(100)
                return CLIP(16, 224, 1, 128, 32, 32, 49408, 128, 2, 1).float().eval().requires_grad_(False)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); split = root / 'compositional-split-natural'; split.mkdir()
            pairs = {'train': [('blue', 'car'), ('red', 'boat'), ('red', 'car')],
                     'val': [('blue', 'boat')], 'test': [('blue', 'boat')]}
            metadata = []; (root / 'images').mkdir()
            for phase in pairs:
                (split / f'{phase}_pairs.txt').write_text(''.join(f'{a} {o}\n' for a, o in pairs[phase]))
                images = pairs[phase] if phase == 'train' else [('blue', 'car'), ('blue', 'boat')]
                for i, (a, o) in enumerate(images):
                    name = f'{phase}_{i}.png'
                    Image.new('RGB', (240, 250), (30 + 30 * i, 80, 130)).save(root / 'images' / name)
                    metadata.append(dict(image=name, attr=a, obj=o, set=phase))
            torch.save(metadata, root / 'metadata_compositional-split-natural.t7')
            self.assertEqual(len(CompositionData(root).records['train']), 3)
            common = dict(data_root=str(root), clip='unused', device='cpu', batch_size=3, workers=0, threads=2)
            bank = str(root / 'support.pt')
            with patch('sba.prepare.load_clip', tiny_clip):
                prepare.run(argparse.Namespace(**common, output=bank))
            config = dict(**common, bank=bank, seed=42, gain=.1, image_checkpoint=False, resume=None)
            with patch('sba.train.load_clip', tiny_clip):
                train.run(argparse.Namespace(**config, epochs=2, output=str(root / 'full')))
                train.run(argparse.Namespace(**config, epochs=1, output=str(root / 'resume')))
                config['resume'] = str(root / 'resume' / 'last.pt')
                train.run(argparse.Namespace(**config, epochs=2, output=str(root / 'resume')))
            full = torch.load(root / 'full/last.pt', weights_only=True)
            resumed = torch.load(root / 'resume/last.pt', weights_only=True)
            for key in full['adapters']:
                torch.testing.assert_close(full['adapters'][key], resumed['adapters'][key], rtol=0, atol=0)
            self.assertEqual(full['history'], resumed['history'])
            with patch('sba.evaluate.load_clip', tiny_clip):
                for setting in ('closed', 'open'):
                    evaluate.run(argparse.Namespace(**common, bank=bank, checkpoint=str(root / 'full/best.pt'),
                        gain=None, phase='test', setting=setting, feasibility=None, output=str(root / setting)))
                    self.assertTrue((root / setting / 'report.json').exists())
                feasibility = torch.tensor([.5, 0., 0., 0.])
                torch.save({'feasibility': feasibility}, root / 'feasibility.pt')
                evaluate.run(argparse.Namespace(**common, bank=bank, checkpoint=str(root / 'full/best.pt'),
                    gain=None, phase='test', setting='open', feasibility=str(root / 'feasibility.pt'),
                    output=str(root / 'filtered')))
                self.assertTrue((root / 'filtered' / 'selected_threshold.json').exists())


if __name__ == '__main__':
    unittest.main()

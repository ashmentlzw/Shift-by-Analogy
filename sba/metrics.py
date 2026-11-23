"""CPU sufficient statistics for the frozen CAMS top-1 evaluation protocol.

Evaluation-only. Preserves float32 bias arithmetic, full-candidate topk tie
resolution, the data-dependent approximately-20-point grid and terminal 1000.
Only the official unweighted evaluation grid is cached. No model
output correction or choice of a deployment bias is performed here.
"""
import numpy as np
from scipy.stats import hmean
import torch


class Curve:
    def __init__(self, scores, truth, train_rows, closed_mask):
        if (scores.device.type != 'cpu' or scores.dtype != torch.float32
                or scores.ndim != 2 or not torch.isfinite(scores).all()
                or scores.abs().max() > 1e6):
            raise ValueError('Finite bounded CPU float32 scores required')
        n, k = scores.shape
        if (truth.dtype != torch.int64 or truth.shape != (n,) or truth.device.type != 'cpu'
                or closed_mask.dtype != torch.bool or closed_mask.shape != (k,)
                or train_rows.dtype != torch.int64 or train_rows.ndim != 1
                or (truth < 0).any() or (truth >= k).any()
                or (train_rows < 0).any() or (train_rows >= k).any()
                or len(torch.unique(train_rows)) != len(train_rows)):
            raise ValueError('Invalid labels/candidate masks')
        seen_candidates = torch.zeros(k, dtype=torch.bool)
        seen_candidates[train_rows] = True
        if not closed_mask[truth].all() or not closed_mask[train_rows].all():
            raise ValueError('Ground truth and every training pair must be closed-world candidates')
        unseen_candidates = closed_mask & ~seen_candidates
        self.seen = seen_candidates[truth].numpy().copy()
        if not self.seen.any() or self.seen.all() or not unseen_candidates.any():
            raise ValueError('Seen and unseen evaluation examples/candidates required')
        self.n = n
        self.fallback_row_bias_evaluations = 0

        def group_top(mask):
            columns = mask.nonzero().flatten()
            values, local = scores[:, columns].topk(min(2, len(columns)), dim=1)
            second = values[:, 1] if len(columns) > 1 else torch.full((n,), -torch.inf)
            return values[:, 0], second, columns[local[:, 0]]

        seen_max, seen_second, seen_index = group_top(seen_candidates)
        unseen_max, unseen_second, unseen_index = group_top(unseen_candidates)

        def correctness(biases):
            result = []
            for biases_chunk in biases.split(32):
                adjusted = unseen_max[:, None] + biases_chunk[None, :]
                adjusted_second = unseen_second[:, None] + biases_chunk[None, :]
                prefer_unseen = adjusted > seen_max[:, None]
                winner = torch.where(prefer_unseen, unseen_index[:, None], seen_index[:, None])
                ambiguous = (adjusted == seen_max[:, None]) | torch.where(
                    prefer_unseen, adjusted == adjusted_second,
                    (seen_max == seen_second)[:, None])
                rr, cc = ambiguous.nonzero(as_tuple=True)
                self.fallback_row_bias_evaluations += len(rr)
                # A uniform float32 addition can create within-group ties.
                # Full K-column topk is necessary; a different tie rule is wrong.
                for start in range(0, len(rr), 128):
                    rows, cols = rr[start:start + 128], cc[start:start + 128]
                    exact = scores[rows].clone()
                    exact[:, ~seen_candidates] += biases_chunk[cols, None]
                    exact[:, ~closed_mask] = -1e10
                    winner[rows, cols] = exact.topk(1, dim=1).indices[:, 0]
                result.append((winner == truth[:, None]).T.to(torch.uint8).numpy())
            return np.concatenate(result, axis=0)

        terminal = torch.tensor([1000.], dtype=torch.float32)
        terminal_correct = correctness(terminal)[0].astype(bool)
        threshold_rows = torch.from_numpy(np.flatnonzero(~self.seen & terminal_correct))
        difference = seen_max[threshold_rows] - scores[threshold_rows, truth[threshold_rows]]
        difference = difference - 1e-4
        order = torch.argsort(difference)
        self.threshold_rows = threshold_rows[order].numpy()
        self.threshold_values = difference[order].numpy()
        all_biases = torch.unique(torch.cat([difference[order][::max(len(difference)//20, 1)], terminal]), sorted=True)
        if n * len(all_biases) > 300_000_000:
            raise ValueError('Curve correctness cache exceeds the explicit 300 MB limit')
        self.biases = all_biases.numpy()
        self.correct = correctness(all_biases)
        self.raw_correct = (scores.masked_fill(~closed_mask, -torch.inf).argmax(-1) == truth).numpy()

    def evaluate(self, weights=None):
        if weights is not None:
            raise ValueError('This evaluator implements the unweighted official curve only')
        weights = np.ones(self.n, dtype=np.int64)
        weights = np.asarray(weights)
        if (weights.shape != (self.n,) or weights.dtype != np.int64
                or (weights < 0).any() or weights.sum() >= 2**24):
            raise ValueError('Nonnegative int64 row multiplicities with total < 2**24 required')
        seen_total, unseen_total = int(weights[self.seen].sum()), int(weights[~self.seen].sum())
        if min(seen_total, unseen_total) <= 0:
            raise ValueError('Resample must contain both seen and unseen examples')
        cumulative = np.cumsum(weights[self.threshold_rows])
        total_correct_unseen = int(cumulative[-1]) if len(cumulative) else 0
        skip = max(total_correct_unseen // 20, 1)
        ranks = np.arange(0, total_correct_unseen, skip, dtype=np.int64)
        locations = np.searchsorted(cumulative, ranks, side='right')
        selected_biases = np.concatenate([self.threshold_values[locations], np.array([1000.], dtype=np.float32)])
        selected = np.searchsorted(self.biases, selected_biases)
        hits = self.correct[selected]
        seen_hits = hits[:, self.seen] @ weights[self.seen]
        unseen_hits = hits[:, ~self.seen] @ weights[~self.seen]
        # Upstream means are float32, then Python float / NumPy float64.
        seen_accuracy = (seen_hits.astype(np.float32) / np.float32(seen_total)).astype(np.float64)
        unseen_accuracy = (unseen_hits.astype(np.float32) / np.float32(unseen_total)).astype(np.float64)
        hm = hmean([seen_accuracy, unseen_accuracy], axis=0)
        best = int(np.argmax(hm))
        result = dict(AUC=float(np.trapz(seen_accuracy, unseen_accuracy)),
            best_hm=float(hm[best]), best_seen=float(seen_accuracy.max()),
            best_unseen=float(unseen_accuracy.max()), hm_seen=float(seen_accuracy[best]),
            hm_unseen=float(unseen_accuracy[best]), biasterm=float(selected_biases[best]),
            raw_closed_top1=float(np.float32(self.raw_correct @ weights) / np.float32(weights.sum())))
        return result

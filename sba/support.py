"""Fixed, TRAIN-only displacement supports with target exclusion."""
import torch


@torch.no_grad()
def build_bank(train_pairs, pairs, text, means):
    """text/means contain TRAIN rows only; means are means of unit image vectors."""
    train_pairs = list(map(tuple, train_pairs))
    if len(text) != len(train_pairs) or means.shape != text.shape:
        raise ValueError('Mismatched training support statistics')
    lookup = {p: i for i, p in enumerate(train_pairs)}
    by_object, by_attr = {}, {}
    for i, (a, o) in enumerate(train_pairs):
        by_object.setdefault(o, []).append(i)
        by_attr.setdefault(a, set()).add(o)
    width = max(1, max(map(len, by_object.values())))
    residual = means - text
    cross = torch.zeros(len(pairs), text.shape[1])
    ids = torch.zeros(len(pairs), width, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    available = torch.zeros(len(pairs), dtype=torch.bool)
    object_attrs = {o: {train_pairs[i][0] for i in rows} for o, rows in by_object.items()}
    for j, (a, o) in enumerate(pairs):
        target = lookup.get((a, o), -1)
        local = [i for i in by_object.get(o, []) if i != target]
        ids[j, :len(local)] = torch.tensor(local, dtype=torch.long)
        mask[j, :len(local)] = True
        triples = [(lookup[(a, p)], lookup[(b, p)], lookup[(b, o)])
                   for b in sorted(object_attrs.get(o, set()) - {a})
                   for p in sorted((by_attr.get(a, set()) & by_attr.get(b, set())) - {o})]
        if triples:
            r = residual[torch.tensor(triples)]
            cross[j] = (r[:, 0] - r[:, 1] + r[:, 2]).mean(0)
            available[j] = True
    return dict(text=text, residual=residual, cross=cross, local_ids=ids,
                local_mask=mask, cross_available=available)

# Training and evaluation

This is the standalone SBA implementation: the adapted visual/text encoders,
TRAIN-only displacement bank, cross-object analogy and same-object family
proxies, image-conditioned gate, joint objective, and SAM/AdamW optimization.
All training starts from public OpenAI CLIP ViT-L/14. No separately trained
CZSL model or fitted generative model is required.

Pretrained checkpoints and matching support banks are available in
[PRETRAINED.md](PRETRAINED.md). They can be evaluated without retraining.

## Environment

Python 3.10; the reference environment uses PyTorch 2.6.0 / torchvision 0.21.0.
Run commands from the repository root. One CUDA GPU is supported; select it
with `CUDA_VISIBLE_DEVICES`. Use a GPU with approximately 48 GB memory for the
reference batch size of 64. `--image-checkpoint` lowers activation memory at
the cost of additional computation. No mixed precision is enabled.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Install the appropriate PyTorch wheel for your CUDA driver if it differs.
The OpenAI CLIP dependency is pinned to a source commit. Download the public
`ViT-L-14.pt` checkpoint using OpenAI CLIP's downloader, or supply an existing
copy:

```bash
mkdir -p checkpoints
python -c "import clip; clip.load('ViT-L/14', device='cpu', download_root='checkpoints')"
```

Expected SHA256:
`b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`.
The text context is 32 tokens, retaining the first 32 pretrained positional
embeddings. Overlength prompts raise an error instead of truncating silently.

## Data

Use the standard compositional-split-natural releases of UT-Zappos,
MIT-States, and C-GQA. Obtain the data through the respective dataset owners
or the preparation instructions in [CAMS](https://github.com/ybyangjing/CAMS)
and [Troika](https://github.com/bighuang624/Troika). Dataset files are not
included in this repository. Each root must have:

```text
data/mit-states/
  images/                               # original metadata-relative filenames
  metadata_compositional-split-natural.t7
  compositional-split-natural/
    train_pairs.txt
    val_pairs.txt
    test_pairs.txt
```

The same structure applies to `data/ut-zappos` and `data/cgqa`. Preserve the
image paths in metadata, including MIT-States underscore folders; UT image
symlinks are supported. Missing or corrupt images raise an error rather than
silently replacing them. The reader checks split image identities for overlap.

| Dataset | TRAIN images | VAL images | TEST images | Open candidates |
| --- | ---: | ---: | ---: | ---: |
| UT-Zappos | 22,998 | 3,214 | 2,914 | 192 |
| MIT-States | 30,338 | 10,420 | 12,995 | 28,175 |
| C-GQA | 26,920 | 7,280 | 5,098 | 278,362 |

## 1. Construct the fixed support bank

```bash
python -m sba.prepare \
  --data-root data/mit-states --clip checkpoints/ViT-L-14.pt \
  --output runs/mit/support.pt --device cuda
```

Only TRAIN images are opened. Their frozen public CLIP embeddings are
normalized individually and averaged per seen composition. The displacement
is that mean minus its frozen text embedding; the mean itself is not
renormalized. For each target, every legal three-corner analogy is averaged.
The target composition is excluded from both its analogy supports and its
same-object supports. All Cartesian candidates are prepared; no support cap
or missing-composition subsampling is applied. This bank is fixed during
training. VAL/TEST images never contribute to the bank.

## 2. Train

```bash
python -m sba.train \
  --data-root data/mit-states --clip checkpoints/ViT-L-14.pt \
  --bank runs/mit/support.pt --output runs/mit/train --device cuda
```

Use the same commands with the UT-Zappos/C-GQA data root and distinct output
paths. Defaults for all three datasets are seed 42, batch size 64, 12 epochs,
24 visual and 12 text adapters of rank 64 and residual scale 0.1, proxy width
128, attention width 32, and auxiliary inference gain 0.1. CLIP stays frozen.

The loss is `0.5 * (pair CE + fused CE) + 0.1 * alpha * (proxy CE + attribute CE
+ object CE)`. `alpha` increases linearly from 0 to 1 over epoch 1. Each update
uses two SAM positions (radius 0.05), then one AdamW step with weight decay 0
and gradient clipping at norm 5. The learning rate rises from 1e-5 to 1e-4 in
epoch 1, then follows cosine decay to 1e-5 over the remaining 11 epochs.
Training uses all seen composition candidates for each batch.

`best.pt` is selected by **closed-world VAL AUC**, with the earliest epoch
winning ties. `last.pt` contains the optimizer and the last complete epoch.
Checkpoints are written atomically. Resume in the original output directory:

```bash
python -m sba.train \
  --data-root data/mit-states --clip checkpoints/ViT-L-14.pt \
  --bank runs/mit/support.pt --output runs/mit/train \
  --resume runs/mit/train/last.pt --device cuda
```

Use the same seed, gain, batch size, and image-checkpoint option when resuming.
If interrupted inside an epoch, that epoch is replayed from its beginning;
per-image augmentation and sampling are deterministic. `--epochs 4` stops
after epoch 4 while retaining the original 12-epoch learning-rate schedule.
Training performs VAL evaluation only; TEST is an explicit separate command.

For a process that survives an SSH disconnect:

```bash
nohup python -u -m sba.train \
  --data-root data/mit-states --clip checkpoints/ViT-L-14.pt \
  --bank runs/mit/support.pt --output runs/mit/train \
  > mit-training.log 2>&1 < /dev/null &
```

## 3. Evaluate closed world

```bash
python -m sba.evaluate \
  --data-root data/mit-states --clip checkpoints/ViT-L-14.pt \
  --bank runs/mit/support.pt --checkpoint runs/mit/train/best.pt \
  --setting closed --phase test --output runs/mit/closed-test
```

The scoring matrix retains the sorted official union of pairs. The closed
mask permits TRAIN + TEST compositions for TEST, or TRAIN + VAL for VAL.
S/U/HM/AUC follow the CAMS float32 top-1 bias curve, approximately 20 sampled
biases plus the terminal bias 1000. Exact full-candidate top-k resolves ties.
The JSON reports both fractions and percentages; AUC is multiplied by 100,
not 10,000. S and U are independent maxima along the curve; HM is the maximum
harmonic mean along that curve, not the harmonic mean of those two maxima.

## 4. Evaluate open world

Unfiltered evaluation uses the full sorted attribute × object product:

```bash
python -m sba.evaluate \
  --data-root data/mit-states --clip checkpoints/ViT-L-14.pt \
  --bank runs/mit/support.pt --checkpoint runs/mit/train/best.pt \
  --setting open --phase test --output runs/mit/open-test
```

For **feasibility-filtered** evaluation, supply the corresponding
`feasibility_mit-states.pt`, `feasibility_ut-zappos.pt`, or
`feasibility_cgqa.pt` from [Troika's data directory](https://github.com/bighuang624/Troika/tree/main/code/data):

```bash
python -m sba.evaluate \
  --data-root data/mit-states --clip checkpoints/ViT-L-14.pt \
  --bank runs/mit/support.pt --checkpoint runs/mit/train/best.pt \
  --setting open --phase test --feasibility data/feasibility_mit-states.pt \
  --output runs/mit/open-filtered-test
```

This first evaluates 50 evenly spaced thresholds between the minimum and
maximum unseen feasibility scores on **open VAL**, chooses the highest VAL
AUC (earliest tie), freezes that threshold, then evaluates TEST. The checkpoint
stays fixed. Files must follow sorted attribute × object order. The reproduced
Troika-style multiplier is `(feasibility >= threshold).float() + seen_mask.float()`;
removed scores become zero rather than negative infinity. This distinction
and the seen-score addition are deliberate. Always label filtered and
unfiltered results separately.

Scores are written as disk-backed NumPy arrays. C-GQA open evaluation requires
substantial host memory and disk (roughly 8 GB VAL and 6 GB TEST scores alone,
plus working copies). Its threshold scan may take considerably longer than
closed evaluation. Output directories must be distinct between evaluations.

## Result provenance and optional gain

The default gain is **0.1**, the formal training setting. `--gain` on the
evaluator is an explicit inference-only override and is recorded in the JSON.
The later UT-Zappos gain **0.445** was selected using closed TEST AUC; it is
exploratory TEST tuning, not a VAL-selected hyperparameter. It must not be
reported as a VAL-only selection protocol. The existing README and its result
table are retained from the repository; this code release does not relabel
those historical numbers or claim a new full retraining verification.

No dataset, pretrained weight, trained checkpoint, cached feature bank,
internal experiment log, or credential is distributed. `sba/` contains only
portable runtime code. See `THIRD_PARTY.md` for dependency acknowledgements.

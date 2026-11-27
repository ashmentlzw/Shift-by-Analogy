# Pretrained SBA models

Download the three inference packages from the
[v1.0.0 release](https://github.com/ashmentlzw/Shift-by-Analogy/releases/tag/v1.0.0).
Each model is the checkpoint selected by **closed-world validation AUC** from
its completed training run (seed 42). Parameters are exported in float32,
without quantization or additional training.

| Dataset | Selected epoch | Closed TEST AUC | Unfiltered open TEST AUC | Download |
| --- | ---: | ---: | ---: | --- |
| UT-Zappos | 4 | 44.9132 | 34.6574 | [Model package](https://github.com/ashmentlzw/Shift-by-Analogy/releases/download/v1.0.0/sba-ut-zappos-v1.0.0.tar.gz) |
| MIT-States | 3 | 24.2887 | 8.2973 | [Model package](https://github.com/ashmentlzw/Shift-by-Analogy/releases/download/v1.0.0/sba-mit-states-v1.0.0.tar.gz) |
| C-GQA | 4 | 17.3031 | 4.3542 | [Model package](https://github.com/ashmentlzw/Shift-by-Analogy/releases/download/v1.0.0/sba-cgqa-v1.0.0.tar.gz) |

AUC values are percentages from the original frozen evaluations with gain
**0.1**. Open results above are **unfiltered**. The optional UT inference gain
**0.445** uses the same weights and is an exploratory setting selected using
closed-world validation AUC.

## Package contents

- `best.pt`: all learned visual/text adapters and proxy-network parameters.
- `support.pt`: the matching fixed TRAIN-only support bank, including every
  Cartesian open-world candidate.
- `metadata.json`: selection, metrics, checksums and compatibility checks.
- `README.md`: dataset-specific evaluation command.

The support bank is necessary: its hash is recorded in `best.pt` and checked
by the evaluator. Keep each model and its support bank together. Dataset
images, individual image embeddings, optimizer state, internal logs and the
frozen public CLIP checkpoint are not included. These packages are for
inference; use a full training `last.pt` for optimizer-state resumption.

## Download and evaluate

Install dependencies and obtain public `ViT-L-14.pt` as described in
[RUNNING.md](RUNNING.md). For example, download MIT-States:

```bash
mkdir -p checkpoints
curl -fL --retry 3 -o checkpoints/sba-mit-states-v1.0.0.tar.gz \
  https://github.com/ashmentlzw/Shift-by-Analogy/releases/download/v1.0.0/sba-mit-states-v1.0.0.tar.gz
curl -fL --retry 3 -o checkpoints/SHA256SUMS \
  https://github.com/ashmentlzw/Shift-by-Analogy/releases/download/v1.0.0/SHA256SUMS
(cd checkpoints && sha256sum --ignore-missing -c SHA256SUMS)
tar -xzf checkpoints/sba-mit-states-v1.0.0.tar.gz -C checkpoints

python -m sba.evaluate \
  --data-root data/mit-states --clip checkpoints/ViT-L-14.pt \
  --bank checkpoints/sba-mit-states/support.pt \
  --checkpoint checkpoints/sba-mit-states/best.pt \
  --setting closed --phase test --output runs/mit/released-closed-test
```

For UT-Zappos or C-GQA replace `mit-states` with `ut-zappos` or `cgqa` in the
filenames and data paths. Use `--setting open` and a new output directory for
unfiltered open-world evaluation. For feasibility-filtered evaluation, follow
the separate VAL-threshold selection instructions in [RUNNING.md](RUNNING.md).
To reproduce the exploratory UT gain setting, explicitly pass `--gain 0.445`.

The exported packages have been reloaded with the public runtime. Learned
parameter tensors match the selected original checkpoints exactly. Closed and
open image/text/score probes agree with the frozen experiment outputs within
CPU-versus-GPU floating-point tolerance. This export verification does not
represent a new training run or hyperparameter search.

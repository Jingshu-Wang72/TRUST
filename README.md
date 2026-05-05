# TRUST

Clean PyTorch implementation of TRUST for UCR time-series classification.

This folder is intended for a public GitHub release. It keeps only source code,
configuration, and documentation. The UCR resample files are treated as external
data and are not committed to the repository.

## Data

The expected benchmark is 112 UCR datasets with 30 resamples per dataset. Put
the files under a data root with this layout:

```text
data/UCR112_30Resamples/
  ACSF1/
    ACSF10_TRAIN.ts
    ACSF10_TEST.ts
    ...
    ACSF129_TRAIN.ts
    ACSF129_TEST.ts
  Adiac/
    Adiac0_TRAIN.ts
    Adiac0_TEST.ts
    ...
```

Each dataset folder must contain `<DatasetName><resample_id>_TRAIN.ts` and
`<DatasetName><resample_id>_TEST.ts` for resample ids `0` to `29`.

The dataset is available at：https://drive.google.com/file/d/1V36LSZLAK6FIYRfPx6mmE5euzogcXS83/view

## Install

```bash
conda create -n trust python=3.10 -y
conda activate trust
pip install -r requirements.txt
```

## Run One Dataset

```bash
python -m trust.train \
  --data_root data/UCR112_30Resamples \
  --dataset ACSF1 \
  --resample_id 0 \
  --output_dir results/ACSF1_r0
```

## Run All UCR Resamples

```bash
python -m trust.run_ucr_resamples \
  --data_root data/UCR112_30Resamples \
  --output_dir results/ucr112_resamples \
  --resample_ids 0-29 \
  --expected_datasets 112
```

The runner writes:

- `all_runs.csv`: one row per dataset/resample run.
- `summary.csv`: mean and standard deviation over the requested resamples.

## Reproducibility Notes

- Default views are:
  `raw,slope,trend,residual,delta,curvature,wavelet,frequency_energy,curvature_event,fft,acf,paa,minirocket,multirocket,hydra`.
- TRUST builds per-view evidential classifiers, creates a global pseudo view,
  and fuses the most reliable views with Dempster-Shafer evidence fusion.
- MiniROCKET, MultiROCKET, and HYDRA use the same evidence DNN branch type as
  all other views; there is no separate Rocket-specific classifier head.
- Per-view feature scalers are fitted on the training split only; the test split
  is transformed with the corresponding training scaler.
- Training runs for the configured number of epochs, 200 by default. Test
  accuracy is evaluated once after training and is not used for early stopping,
  learning-rate scheduling, or best-model selection.

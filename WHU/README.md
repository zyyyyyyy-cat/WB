# UAD and token-specialized adapter experiments on one GPU

Edit `DATASETS.ROOT_DIR` and `MODEL.PRETRAIN_PATH` in `configs/UAD.yml` before
running. Apply the same path edits to the three adapter configs. The single-GPU
batch setting (`IMS_PER_BATCH=64`) represents 16 IDs x 4 samples x 3 modalities,
i.e. 192 images per iteration.

The controlled experiments differ only inside each ViT block:

- P0 `UAD.yml`: original UAD.
- P1 `UAD_DPA.yml`: patch-only DPA in parallel with every FFN.
- P2 `UAD_TCA.yml`: CLS-only modulation plus one class adapter shared across layers.
- P3 `UAD_DPA_TCA.yml`: DPA and TCA together.

All four settings retain ImageNet initialization, full backbone fine-tuning,
PKM sampling, 120 epochs, SGD, CE + WRT + 0.01 ProCA + GPD, and the original
global pre-BN CLS descriptor. IFA and all modality/platform-specific branches
are intentionally absent.

At each existing `LOG_PERIOD` iteration, adapter experiments additionally log
`R_DPA = ||DPA(P)|| / (||FFN(P)|| + eps)` and/or
`R_TCA = ||TCA(c)|| / (||FFN(c)|| + eps)`. Squared Frobenius norms are summed
over all enabled transformer blocks before taking the ratio. The statistics
are detached, are not model parameters, and do not participate in the loss or
backward pass. P0 records neither ratio; P1 records DPA; P2 records TCA; P3
records both.

## WHU-MARS-1000

CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/UAD.yml
CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/UAD.yml TEST.WEIGHT logs/UAD_1000/transformer_120.pth

## P1/P2/P3 on WHU-MARS-1000

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/UAD_DPA.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/UAD_TCA.yml
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/UAD_DPA_TCA.yml

CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/UAD_DPA.yml TEST.WEIGHT logs/UAD_DPA_1000/transformer_120.pth
CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/UAD_TCA.yml TEST.WEIGHT logs/UAD_TCA_1000/transformer_120.pth
CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/UAD_DPA_TCA.yml TEST.WEIGHT logs/UAD_DPA_TCA_1000/transformer_120.pth
```

Testing logs the overall AS-ReID metrics, all nine modality query/gallery
pairs, A->G and G->A, and the RGB/NIR/TIR aerial-query slices against the
all-modality ground gallery. The last group contains the targeted `TIR/A->G`
diagnostic.

## P0/P1 layer-wise feature export

The exporter writes chunked PyTorch files containing normalized CLS and patch
states at L3/L6/L9/L12, final descriptors, query/gallery flags, identities,
cameras, modalities, and image paths. `float16` is the storage default to limit
disk usage; computation remains at the model's normal precision.

```bash
CUDA_VISIBLE_DEVICES=0 python export_layer_features.py \
  --config_file configs/UAD.yml \
  --weight logs/UAD_1000/transformer_120.pth \
  --output_dir logs/diagnostics/P0 \
  --layers 3 6 9 12 --subset all --storage_dtype float16 \
  TEST.IMS_PER_BATCH 32

CUDA_VISIBLE_DEVICES=0 python export_layer_features.py \
  --config_file configs/UAD_DPA.yml \
  --weight logs/UAD_DPA_1000/transformer_120.pth \
  --output_dir logs/diagnostics/P1 \
  --layers 3 6 9 12 --subset all --storage_dtype float16 \
  TEST.IMS_PER_BATCH 32
```

Patch tensors are large. Use `--subset query` first for a smaller diagnostic
run, and reduce `TEST.IMS_PER_BATCH` if GPU memory is limited.


## WHU-MARS-1000-GD

CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/UAD.yml DATASETS.FILTER_GD True OUTPUT_DIR logs/UAD_1000_GD TEST.WEIGHT logs/UAD_1000/transformer_120.pth


## WHU-MARS-2337

CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/UAD.yml DATASETS.DATASET_DIR WHU-MARS-2337 OUTPUT_DIR logs/UAD_2337
CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/UAD.yml DATASETS.DATASET_DIR WHU-MARS-2337 OUTPUT_DIR logs/UAD_2337 TEST.WEIGHT logs/UAD_2337/transformer_120.pth

"""Export L3/L6/L9/L12 CLS and patch features for P0/P1 diagnostics."""

import argparse
import json
import os

import torch

from config import cfg
from datasets import make_dataloader
from model import make_model


def parse_args():
    parser = argparse.ArgumentParser(description="Export layer-wise UAD features")
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--weight", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--layers", nargs="+", type=int, default=[3, 6, 9, 12])
    parser.add_argument("--subset", choices=("all", "query", "gallery"), default="all")
    parser.add_argument("--storage_dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument(
        "opts", help="YACS overrides, e.g. TEST.IMS_PER_BATCH 32", nargs=argparse.REMAINDER
    )
    return parser.parse_args()


def select_subset(is_query, subset):
    if subset == "query":
        return is_query
    if subset == "gallery":
        return ~is_query
    return torch.ones_like(is_query, dtype=torch.bool)


def main():
    args = parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", cfg.MODEL.DEVICE_ID)
    if not torch.cuda.is_available():
        raise RuntimeError("Layer feature export requires one CUDA GPU.")
    device = torch.device("cuda:0")
    os.makedirs(args.output_dir, exist_ok=True)

    _, _, val_loaders, num_querys, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(
        cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num
    )
    model.load_param(args.weight)
    model.to(device).eval()

    storage_dtype = torch.float16 if args.storage_dtype == "float16" else torch.float32
    manifest = {
        "config": os.path.abspath(args.config_file),
        "weight": os.path.abspath(args.weight),
        "layers": args.layers,
        "subset": args.subset,
        "storage_dtype": args.storage_dtype,
        "modalities": list(cfg.DATASETS.MODALITIES),
        "chunks": [],
    }

    for modality, loader, num_query in zip(cfg.DATASETS.MODALITIES, val_loaders, num_querys):
        modality_dir = os.path.join(args.output_dir, modality)
        os.makedirs(modality_dir, exist_ok=True)
        sample_offset = 0
        chunk_index = 0

        for img, pid, camid, _, modid, imgpath in loader:
            batch_size = img.shape[0]
            is_query = torch.arange(
                sample_offset, sample_offset + batch_size, dtype=torch.long
            ) < num_query
            keep = select_subset(is_query, args.subset)
            sample_offset += batch_size
            if not keep.any():
                continue

            with torch.no_grad():
                descriptor, layer_features = model.forward_diagnostics(
                    img.to(device), layers=args.layers
                )

            kept_indices = keep.nonzero(as_tuple=False).flatten().tolist()
            payload = {
                "modality": modality,
                "pid": torch.as_tensor(pid, dtype=torch.long)[keep],
                "camid": torch.as_tensor(camid, dtype=torch.long)[keep],
                "modid": torch.as_tensor(modid, dtype=torch.long)[keep],
                "is_query": is_query[keep],
                "imgpath": [imgpath[index] for index in kept_indices],
                "descriptor": descriptor[keep].detach().to("cpu", dtype=storage_dtype),
                "layers": {
                    layer: {
                        "cls": features["cls"][keep].detach().to("cpu", dtype=storage_dtype),
                        "patch": features["patch"][keep].detach().to("cpu", dtype=storage_dtype),
                    }
                    for layer, features in layer_features.items()
                },
            }
            chunk_name = "chunk_{:05d}.pt".format(chunk_index)
            chunk_path = os.path.join(modality_dir, chunk_name)
            torch.save(payload, chunk_path)
            manifest["chunks"].append(os.path.relpath(chunk_path, args.output_dir))
            chunk_index += 1

    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

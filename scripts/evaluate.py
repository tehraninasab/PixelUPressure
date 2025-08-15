#!/usr/bin/env python
# ===============================================================
#  eval_cxr_model.py
#  ---------------------------------------------------------------
#      Evaluate a folder of generated Chest‑X‑ray images against
#      CheXpert validation data.
#      Metrics:  (1) FID (XRV DenseNet‑121)   (2) Vendi Score
# ===============================================================
import argparse, math, warnings, json, os, random
from pathlib import Path
from typing   import List, Dict, Callable
from tqdm import tqdm
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import skimage.io
import torchxrayvision as xrv
from vendi_score import vendi
from torchmetrics.image.fid import FrechetInceptionDistance

TARGET_COLS = ["Cardiomegaly", "Lung Opacity", "Edema", "No Finding", "Pneumothorax", "Pleural Effusion"]

class XRVPathDataset(torch.utils.data.Dataset):
    """Load image → normalize → crop → resize(224) → tensor[-1024,1024]."""
    def __init__(self, paths: List[Path]):
        self.paths = list(map(str, paths))
        self.crop_resize = transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224)
        ])

    def __len__(self): 
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = skimage.io.imread(self.paths[idx].replace("CheXpert-v1.0/", "CheXpert-v1.0_1024x1024/"))
        img = xrv.datasets.normalize(img, 255)
        if img.ndim == 2:              
            img = img[None]
        elif img.shape[-1] == 3:      
            img = img.mean(2, keepdims=True).transpose(2,0,1)
        elif img.shape[0] == 1:        
            pass
        img = self.crop_resize(img)
        return torch.from_numpy(img)

def make_loader(paths, bs, nw):
    return DataLoader(XRVPathDataset(paths), bs, False,
                      num_workers=nw, pin_memory=True)

def dense121_feature_extractor(device="cpu") -> nn.Module:
    base = xrv.models.DenseNet(weights="densenet121-res224-all").to(device).eval()
    class Feat(nn.Module):
        num_features = 1024
        def forward(self, x):
            if x.size(1) == 3: x = x.mean(1, keepdim=True)
            z = F.relu(base.features(x), inplace=True)
            return F.adaptive_avg_pool2d(z, 1).flatten(1)
    return Feat()

@torch.inference_mode()
def fid_score(real_paths, fake_paths, batch, device="cuda") -> float:
    feat = dense121_feature_extractor(device)
    metric = FrechetInceptionDistance(
        feature=feat, normalize=True, reset_real_features=True,
        input_img_size=(1,224,224)
    ).to(device)
    dl_r = make_loader(real_paths, batch, 4)
    dl_f = make_loader(fake_paths, batch, 4)
    for xb in tqdm(dl_r): metric.update(xb.to(device), real=True)
    for xb in tqdm(dl_f): metric.update(xb.to(device), real=False)
    return float(metric.compute())

@torch.no_grad()
def extract_latents(paths, batch, device="cuda") -> np.ndarray:
    feat = dense121_feature_extractor(device)
    loader = make_loader(paths, batch, 4)
    out = []
    for xb in loader:
        out.append(feat(xb.to(device)).cpu())
    return torch.cat(out,0).numpy().astype("float32") 

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chexpert_csv",  required=True)
    ap.add_argument("--chexpert_root", required=True)
    ap.add_argument("--gen_csv",     required=True)
    ap.add_argument("--gen_root",      required=True,
                    help="Path to generated images")
    ap.add_argument("--out_dir",       default="eval_out")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model_name = Path(args.gen_root).stem          # e.g.  logs‑21001 → logs‑21001
    out_dir = Path(args.out_dir)
    lat_dir = out_dir / f"latents-{model_name}" / "dense224"
    lat_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    df_meta = pd.read_csv(args.chexpert_csv)
    fake_df = pd.read_csv(args.gen_csv)
    ss = 5000
    subset_sizes = [ss]

    real_all:list[str] = []          # accumulate real paths
    fake_all:list[str] = []          # accumulate every fake img
    subset_pool: dict[int, list[np.ndarray]] = {n: [] for n in subset_sizes}

    for p_idx, condition in enumerate(TARGET_COLS):
        
        tag = f"{p_idx:02d}"
        mask = df_meta[condition] == 1
        real_paths = [os.path.join(args.chexpert_root, p) for p in df_meta.loc[mask, "Path"]]
        if not real_paths:
            warnings.warn(f"No real images found for condition: {condition}")
            continue

        if len(real_paths) > ss:
            real_all.extend(real_paths[:ss])
        else:
            real_all.extend(real_paths)

        fake_mask = fake_df[condition] == 1
        fake_paths = [os.path.join(args.gen_root, p) for p in fake_df.loc[fake_mask, "Path"]]

        if not fake_paths:
            warnings.warn(f"No generated images for condition: {condition}")
            continue

        if len(real_paths) < ss:
            fake_all.extend(fake_paths[:len(real_paths)])
        else:
            fake_all.extend(fake_paths)

        # ---------- FID (condition) -------------------------------------------------
        fid_val = fid_score(real_paths, fake_paths, args.batch, args.device)
        print(f"\nFID({condition}): {fid_val}")
        results.append(dict(condition=condition, subset="ALL", metric="FID", value=fid_val))

        # ---------- Latents (once) ----------------------------------------------
        z_full = extract_latents(fake_paths, args.batch, args.device)      # (K,1024)

        # save full latent set for this condition
        np.save(lat_dir / f"{tag}_full.npy", z_full)

        # ---------- subset handling / Vendi -------------------------------------
        for n in subset_sizes:
            if len(z_full) < n:
                continue
            z_sub = z_full[:n]
            np.save(lat_dir / f"{tag}_{n}.npy", z_sub)                     # cache
            subset_pool[n].append(z_sub)                                   # for ALL
            if z_sub.shape[0] < n:
                vendi_val = float(vendi.score(z_sub))
            else:
                vendi_val = float(vendi.score_dual(z_sub))

            print(f"Vendi({condition})-{n}: {vendi_val}")
            results.append(dict(condition=condition, subset=str(n),
                                metric="Vendi", value=vendi_val))

    # ======================  GLOBAL ("ALL") METRICS  =============================
    # 1)  FID across the whole validation split
    print("lenghts: ", len(real_all), len(fake_all))
    fid_all = fid_score(real_all, fake_all, args.batch, args.device)
    print(f"\nFID (all conditions): {fid_all}")
    results.append(dict(condition="ALL", subset="ALL", metric="FID", value=fid_all))

    # 2)  Vendi for each subset size aggregated across conditions
    for n in subset_sizes:
        if not subset_pool[n]:
            continue
        z_concat = np.concatenate(subset_pool[n], axis=0)   # (P*n, 1024)
        np.save(lat_dir / f"ALL_{n}.npy", z_concat)         # cache for later use
        vendi_all = float(vendi.score_dual(z_concat))
        print(f"Vendi(all, first {n}/condition): {vendi_all}")
        results.append(dict(condition="ALL", subset=str(n),
                            metric="Vendi", value=vendi_all))

    res_df = pd.DataFrame(results)
    csv_path = out_dir / f"results-{model_name}.csv"
    res_df.to_csv(csv_path, index=False)

    print(f"\nFID (all conditions): {fid_all:6.2f}")
    for n in subset_sizes:
        if subset_pool[n]:
            print(f"Vendi(all, first {n}/condition): {res_df[(res_df.condition=='ALL')&(res_df.subset==str(n))].value.iloc[0]:.4f}")
    print(f"Latents saved to {lat_dir}")
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()

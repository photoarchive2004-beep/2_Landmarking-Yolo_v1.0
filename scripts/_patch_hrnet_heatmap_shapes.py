from pathlib import Path

root = Path(__file__).resolve().parent
train_path = root / "scripts" / "train_hrnet.py"

text = train_path.read_text(encoding="utf-8")

old_block = """            _, _, H, W = imgs.shape
            gt_heatmaps = keypoints_to_heatmaps(
                kps,
                H,
                W,
                sigma=float(getattr(cfg, "heatmap_sigma_px", 2.0)),
                device=device,
            )
            pred_heatmaps = model(imgs)
            loss = criterion(pred_heatmaps, gt_heatmaps)
"""

new_block = """            pred_heatmaps = model(imgs)
            _, _, H_hm, W_hm = pred_heatmaps.shape
            gt_heatmaps = keypoints_to_heatmaps(
                kps,
                H_hm,
                W_hm,
                sigma=float(getattr(cfg, "heatmap_sigma_px", 2.0)),
                device=device,
            )
            loss = criterion(pred_heatmaps, gt_heatmaps)
"""

if old_block not in text:
    print("[ERR] Target training block not found in train_hrnet.py, patch not applied.")
else:
    text = text.replace(old_block, new_block)
    train_path.write_text(text, encoding="utf-8")
    print("[INFO] train_hrnet.py patched: gt_heatmaps now use model heatmap size.")

from pathlib import Path

root = Path(__file__).resolve().parent.parent
train_path = root / "scripts" / "train_hrnet.py"
print("[INFO] train_hrnet.py:", train_path)

text = train_path.read_text(encoding="utf-8")

old_train_block = """            optimizer.zero_grad()
            _, _, H, W = imgs.shape
            gt_heatmaps = keypoints_to_heatmaps(
                kps,
                H,
                W,
                sigma=float(getattr(cfg, "heatmap_sigma_px", 2.0)),
                device=device,
            )
            pred_heatmaps = model(imgs)
            loss = criterion(pred_heatmaps, gt_heatmaps)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1
"""

new_train_block = """            optimizer.zero_grad()
            # Сначала прогоняем через модель и узнаём размер теплокарт
            pred_heatmaps = model(imgs)
            _, _, H_hm, W_hm = pred_heatmaps.shape
            _, _, H_in, W_in = imgs.shape

            # Масштабируем ключевые точки под разрешение теплокарт,
            # чтобы gt и pred были одного размера.
            if H_in != H_hm or W_in != W_hm:
                scale_x = float(W_hm) / float(W_in)
                scale_y = float(H_hm) / float(H_in)
                kps_scaled = kps.clone()
                kps_scaled[..., 0] = kps_scaled[..., 0] * scale_x
                kps_scaled[..., 1] = kps_scaled[..., 1] * scale_y
            else:
                kps_scaled = kps

            gt_heatmaps = keypoints_to_heatmaps(
                kps_scaled,
                H_hm,
                W_hm,
                sigma=float(getattr(cfg, "heatmap_sigma_px", 2.0)),
                device=device,
            )
            loss = criterion(pred_heatmaps, gt_heatmaps)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1
"""

if old_train_block not in text:
    print("[ERR] Training inner block (optimizer/gt_heatmaps) not found, patch NOT applied.")
else:
    text = text.replace(old_train_block, new_train_block)
    print("[INFO] Training block patched (heatmaps now at HRNet output resolution).")

old_val_block = """        model.eval()
        all_pred = []
        all_gt = []
        with torch.no_grad():
            for imgs, kps in val_loader:
                imgs = imgs.to(device)
                kps = kps.to(device)
                pred_heatmaps = model(imgs)
                pred_kps = heatmaps_to_keypoints(pred_heatmaps)
                all_pred.append(pred_kps.cpu())
                all_gt.append(kps.cpu())
"""

new_val_block = """        model.eval()
        all_pred = []
        all_gt = []
        with torch.no_grad():
            for imgs, kps in val_loader:
                imgs = imgs.to(device)
                kps = kps.to(device)
                pred_heatmaps = model(imgs)
                pred_kps = heatmaps_to_keypoints(pred_heatmaps)
                # Переводим предсказанные координаты в ту же систему, что и gt
                _, _, H_in, W_in = imgs.shape
                _, _, H_hm, W_hm = pred_heatmaps.shape
                if H_in != H_hm or W_in != W_hm:
                    scale_x = float(W_in) / float(W_hm)
                    scale_y = float(H_in) / float(H_hm)
                    pred_kps[..., 0] = pred_kps[..., 0] * scale_x
                    pred_kps[..., 1] = pred_kps[..., 1] * scale_y
                all_pred.append(pred_kps.cpu())
                all_gt.append(kps.cpu())
"""

if old_val_block not in text:
    print("[ERR] Validation block not found, patch for PCK NOT applied.")
else:
    text = text.replace(old_val_block, new_val_block)
    print("[INFO] Validation block patched (PCK computed in input image coordinates).")

train_path.write_text(text, encoding="utf-8")
print("[INFO] train_hrnet.py updated successfully.")

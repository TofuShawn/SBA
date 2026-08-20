# ROCm / HIP on WSL2 (AMD GPU) / WSL2 上的 ROCm/HIP

## Status / 現況
- GPU：**AMD Radeon RX 7900 XTX**（gfx1100）。
- Ubuntu **24.04** WSL 發行版已安裝；ROCm **6.2.4**（`amdgpu-install --usecase=rocm --no-dkms`）與 **PyTorch 2.5.1+rocm6.2**（venv `/opt/sba`，含 numpy）已就緒。
- `alphazero.py` 已內建 GPU 自動切換（`DEVICE = cuda if torch.cuda.is_available()`），無需改碼。

## Blocker / 目前卡點
WSL 內 **`/dev/kfd` 不存在**（`rocminfo` 回報 `ROCk module is NOT loaded`），因此 `torch.cuda.is_available()` 為 False。
這是 **Windows 端 AMD 驅動**沒有暴露 WSL compute 節點所致（目前驅動 `32.0.31035.1003` 偏舊）。

## Fix / 解法（Windows 端手動）
1. 到 [AMD 驅動下載](https://www.amd.com/en/support/download/drivers.html) 安裝支援 **WSL** 的最新 **Adrenalin Edition** 驅動（安裝後可能需重開機）。
2. 重啟 WSL：
   ```powershell
   wsl --shutdown
   ```
3. 驗證 `/dev/kfd` 出現：
   ```powershell
   wsl -d Ubuntu --user root -- ls /dev/kfd
   ```
4. 驗證 PyTorch 看到 GPU：
   ```powershell
   wsl -d Ubuntu --user root -- /opt/sba/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

## Usage / 用法
```powershell
# 在 Ubuntu 內跑 AlphaZero 訓練（repo 掛載於 /mnt/e/Project/SBA）
wsl -d Ubuntu --user root -- sh -c "cd /mnt/e/Project/SBA && /opt/sba/bin/python alphazero.py train --games 400 --sims 80"
```

## Notes / 備註
- `/dev/dri/renderD128` 已存在（顯示/渲染透傳正常）；ROCm 計算另需 `/dev/kfd`。
- 若驅動更新後仍無 `/dev/kfd`，執行 `wsl --update` 再 `wsl --shutdown` 一次。

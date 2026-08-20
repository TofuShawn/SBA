# ROCm / HIP on WSL2 (AMD GPU) / WSL2 上的 ROCm/HIP

## Status / 現況
- GPU：**AMD Radeon RX 7900 XTX**（gfx1100）。
- **原生 Windows**：AMD 官方維護的 **TheRock** nightly 索引提供 `gfx110X-dgpu` 家族的
  **Windows `win_amd64` PyTorch wheel**（`torch` / `torchvision` / `torchaudio`，cp311 / cp312 / cp313）——
  不需要 WSL2、也不需要自行編譯。
- WSL2 Ubuntu **24.04** 已安裝；ROCm **6.2.4**（`amdgpu-install --usecase=rocm --no-dkms`）與
  **PyTorch 2.5.1+rocm6.2**（venv `/opt/sba`，含 numpy）已就緒，可作 fallback。
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

## Native Windows: TheRock index / 原生 Windows：TheRock 索引
- `pytorch.org` 官方索引確實沒有 Windows 的 ROCm wheel；但 **AMD 官方維護的 TheRock** 索引
  （`https://rocm.nightlies.amd.com/v2/<gfx家族>/`）有發佈 Windows wheel。
- RX 7900 XTX（gfx1100）屬於 **`gfx110X-dgpu`**（gfx1100 / gfx1101 / gfx1102，獨立顯示卡；
  `gfx110X-all` 另含 gfx1103 內顯）。安裝（nightly/rc 版需 `--pre`）：
  ```powershell
  pip install --pre --index-url https://rocm.nightlies.amd.com/v2/gfx110X-dgpu/ torch torchvision torchaudio
  ```
- TheRock 新版統一 **multi-arch** 索引（`https://rocm.nightlies.amd.com/whl-multi-arch/`，
  RELEASES.md 建議方式，Windows 的 PyTorch 為 ✅）：
  ```powershell
  pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ "torch[device-gfx1100]" "torchvision[device-gfx1100]" torchaudio
  ```
- 驗證 GPU 可用：
  ```powershell
  python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  ```
- 注意：TheRock 是 **nightly / rc 建置**（如 `2.10.0a0+rocm7.10.0a…`，以 PyTorch main 分支為主），
  穩定性不如正式版；若遇到問題可退回 WSL2 路徑。

## Why WSL2 (fallback) / WSL2 作為備案
- WSL2 仍是 AMD 官方支援的完整路徑，且本機已建好（Ubuntu 24.04 + ROCm 6.2.4 + PyTorch 2.5.1+rocm6.2）。
- 使用 WSL2 時只需更新 Windows 端 AMD 驅動讓 `/dev/kfd` 出現（見上方 Fix / 解法一節）。

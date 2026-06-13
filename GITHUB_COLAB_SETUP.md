# 🚀 GitHub + Google Colab Workflow for TimesFM Training

## ✅ Quy Trình TỰ ĐỘNG - Không Cần Manual Upload!

### 📋 Tổng Quan Workflow:

```
Local Machine (Windows)          GitHub                    Google Colab (T4 GPU)
      │                              │                              │
      │  1. Push code & data         │                              │
      ├─────────────────────────────>│                              │
      │                              │  2. Clone repo               │
      │                              ├─────────────────────────────>│
      │                              │                              │
      │                              │                              │  3. Train TimesFM
      │                              │                              │  (3 hours, T4 16GB)
      │                              │                              │
      │                              │                              │  4. Push results
      │                              │<─────────────────────────────┤
      │                              │                              │
      │  5. Pull trained model       │                              │
      │<─────────────────────────────┤                              │
      │                              │                              │
```

---

## 🎯 BƯỚC 1: Push Lên GitHub (One-time Setup)

### 1.1 Tạo Repository trên GitHub

1. **Đăng nhập GitHub** → [github.com](https://github.com)
2. **Create new repository:**
   - Name: `stockvoli-research`
   - Description: `TimesFM 2.5 Training for Vietnamese VN30 Stocks`
   - **Public** hoặc **Private** (tùy bạn)
   - **KHÔNG** check "Initialize with README" (đã có code sẵn)

### 1.2 Push Local Code Lên GitHub

```bash
# Đã sẵn sàng! Git repo đã được tạo với commit đầu tiên
# Chỉ cần push lên GitHub

# Thêm remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/stockvoli-research.git

# Push lên GitHub
git push -u origin master
```

**Expected Output:**
```
Enumerating objects: 39, done.
Counting objects: 100% (39/39), done.
...
To https://github.com/YOUR_USERNAME/stockvoli-research.git
 * [new branch]      master -> master
```

### 1.3 Verify trên GitHub

1. Mở: `https://github.com/YOUR_USERNAME/stockvoli-research`
2. **Check các files này có trên GitHub:**
   - ✅ `data/processed/` (30 files CSV)
   - ✅ `configs/config.yaml`
   - ✅ `src/model_training_fixed.py`
   - ✅ `colab_git_training.ipynb`
   - ✅ `.gitignore`

**Total size:** ~16 MB (hoàn toàn OK cho GitHub)

---

## 🎯 BƯỚC 2: Setup Google Colab với Git

### 2.1 Mở Notebook trong Colab

1. **Mở Colab** → [colab.google.com](https://colab.google.com)
2. **Upload notebook:** File → Upload → Select `colab_git_training.ipynb`
3. **Connect T4 GPU:** Runtime → Change runtime type → T4 GPU
4. **Save:** File → Save to Google Drive (optional)

### 2.2 Configure GitHub Credentials

**Trong Colab, Cell "Step 3":**

```python
GITHUB_USERNAME = "YOUR_USERNAME"  # @param {type:"string"}
REPO_NAME = "stockvoli-research"
```

**Trong Cell "Step 5":**

```python
# Tạo GitHub Personal Access Token (PAT)
# 1. Đi đến: https://github.com/settings/tokens
# 2. Click "Generate new token" → "Generate new token (classic)"
# 3. Name: "Colab Training"
# 4. Scopes: check "repo" (full control)
# 5. Click "Generate token"
# 6. Copy token (chỉ hiện 1 lần!)

GITHUB_PAT = "ghp_xxxxxxxxxxxx"  # Paste your token here
```

⚠️ **Security Note:** PAT này chỉ dùng trong Colab session, không được lưu lại!

---

## 🎯 BƯỚC 3: Train TimesFM trên Colab (Tự Động)

### 3.1 Clone và Train

**Chạy tất cả cells:**
```
Runtime → Run all
```

**Điều gì sẽ xảy ra:**
1. ✅ Check GPU (T4 16GB)
2. ✅ Install dependencies (transformers, peft, torch)
3. ✅ Clone repo từ GitHub (tự động!)
4. ✅ Load VN30 data (30 stocks)
5. ✅ Train TimesFM 2.5 (~3 hours)
6. ✅ Save models local trong Colab

### 3.2 Theo Dõi Training

**Monitor GPU:**
```python
!nvidia-smi -l 2  # Update mỗi 2 seconds
```

**Xem logs:**
```python
!tail -f experiments/model_training.log
```

**Expected metrics:**
- GPU utilization: 80-90%
- R² improving: 0.0 → 0.5+
- Loss decreasing: 2.0 → 0.5

---

## 🎯 BƯỚC 4: Push Results về GitHub (Tự Động)

### 4.1 Khi Training Hoàn Thành

**Sau khi training xong (~3 hours), Colab sẽ tự động:**

1. Add trained models:
```bash
!git add models/ experiments/
```

2. Commit với metrics:
```bash
!git commit -m "feat: Add trained TimesFM 2.5 model

- Training completed on Google Colab T4 GPU
- R²: 0.55
- QLIKE: 0.85
- RMSE: 0.42"
```

3. Push về GitHub:
```bash
!git push origin master
```

### 4.2 Verify Results trên GitHub

1. Refresh GitHub repo
2. **Check new files:**
   - ✅ `models/best_model_r2_0.XXX.pt`
   - ✅ `experiments/training_metrics.json`
   - ✅ `experiments/model_training.log`

---

## 🎯 BƯỚC 5: Pull Model Về Local (Optional)

### 5.1 Pull về Local Machine

```bash
# Pull models từ GitHub
git pull origin master

# Models giờ available local
ls -lh models/
# Output: best_model_r2_0.XXX.pt
```

### 5.2 Use Model cho Inference

```python
import torch

# Load model đã train
model = torch.load('models/best_model_r2_0.55.pt')
model.eval()

# Dự đoán volatility cho stocks mới
predictions = model.predict(new_data)
```

---

## 🔄 CÁCH DÙNG HỆ THỐNG GITHUB + COLAB

### Scenario 1: Train Lần Đầu

```
Local → GitHub → Colab (train) → GitHub → Local
   1. Push      2. Clone      3. Train      4. Push     5. Pull
```

### Scenario 2: Update Code/Config

```bash
# 1. Sửa code local
vim src/model_training_fixed.py

# 2. Commit & push
git add src/
git commit -m "fix: Improve gradient clipping"
git push origin master

# 3. Trong Colab: Pull changes
!git pull origin master

# 4. Train lại
Runtime → Run all
```

### Scenario 3: Multiple Training Runs

```bash
# Mỗi training run tạo commit mới
# Run 1: "feat: Initial training - R² 0.50"
# Run 2: "feat: Hyperparameter tuning - R² 0.55"
# Run 3: "feat: More epochs - R² 0.60"

git log --oneline  # Xem tất cả training runs
```

---

## 🎯 ƯU ĐIỂM CỦA GITHUB WORKFLOW

### ✅ So với Google Drive Manual Upload:

| Feature | Google Drive | GitHub + Colab |
|---------|--------------|----------------|
| Upload files | Manual (33 files) | Auto (git push) |
| Version control | ❌ No | ✅ Full history |
| Collaboration | ❌ Hard | ✅ Easy |
| Code changes | Manual re-upload | `git pull` |
| Training reproducibility | ❌ Difficult | ✅ Exact (commits) |
| Storage limit | 15GB free | 100MB files, 1GB repo |
| Backup | Manual | Auto |

### ✅ Benefits:

1. **Tự động hóa:** Không cần manual upload
2. **Version tracking:** Mỗi training run là 1 commit
3. **Reproducibility:** Exact code + data + config
4. **Collaboration:** Dễ share với team
5. **Backup:** GitHub backup everything
6. **CI/CD ready:** Dễ integrate với automated pipelines

---

## 🚨 TROUBLESHOOTING

### Problem: "Repository not found"

**Solution:**
1. Check repo URL: `https://github.com/YOUR_USERNAME/stockvoli-research`
2. Make sure repo is **Public** hoặc you have access
3. Verify PAT has `repo` scope

### Problem: "Authentication failed"

**Solution:**
1. Generate new PAT: https://github.com/settings/tokens
2. Make sure PAT has `repo` scope
3. Update in Colab: `GITHUB_PAT = "new_token"`

### Problem: "Data files too large"

**Solution:**
Git LFS (Large File Storage):
```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "data/processed/*.csv"
git add .gitattributes
git commit -m "chore: Add Git LFS tracking"
```

### Problem: "Colab session timeout"

**Solution:**
1. Colab timeout sau 90min inactive
2. Training auto-saves checkpoints
3. Reconnect & continue training

---

## 📊 EXPECTED TIMELINE

### First-time Setup (One-time):
- Create GitHub repo: 2 min
- Push local code: 5 min
- Total setup: **7 min**

### Each Training Run:
- Clone in Colab: 1 min
- Setup GPU & deps: 5 min
- Load data: 2 min
- **Training: ~3 hours**
- Push results: 2 min
- **Total: ~3 hours 10 min**

### Future Updates:
- Modify code: 5 min
- Push to GitHub: 1 min
- Pull in Colab: 1 min
- Retrain: ~3 hours
- **Total: ~3 hours 7 min**

---

## 🎯 SUCCESS CRITERIA

✅ **Setup Complete When:**
- GitHub repo has 36 files (30 data + 6 code/config)
- Colab can clone repo successfully
- GPU shows 16GB memory

✅ **Training Complete When:**
- `models/best_model_r2_0.XXX.pt` exists
- R² > 0.5
- QLIKE < 1.0
- No CUDA OOM errors

✅ **Workflow Complete When:**
- Models pushed to GitHub
- Local machine can pull models
- Model can load for inference

---

## 🎓 BẮT ĐẦU NGAY!

### Quick Start Commands:

```bash
# 1. Tạo GitHub repo (manual trên web)
# https://github.com/new

# 2. Push local code
cd D:/bmad-projects/stockvoli-research
git remote add origin https://github.com/YOUR_USERNAME/stockvoli-research.git
git push -u origin master

# 3. Mở Colab với notebook
# File → Upload → colab_git_training.ipynb

# 4. Configure GitHub credentials
GITHUB_USERNAME = "YOUR_USERNAME"
GITHUB_PAT = "your_token_here"

# 5. Run all cells
Runtime → Run all

# 6. Wait ~3 hours
# Results tự động push về GitHub!

# 7. Pull về local
git pull origin master
ls models/  # Your trained model!
```

---

## 🎯 SUMMARY

**Bạn Đã Sẵn Sàng!**

1. ✅ Git repo đã được tạo
2. ✅ Code & data đã commit
3. ✅ Colab notebook đã sẵn sàng
4. ✅ Documentation hoàn chỉnh

**Chỉ Cần:**
1. Create GitHub repo
2. Push code lên
3. Mở Colab & train
4. Get trained model!

**Total time:** ~3 hours 10 min (chỉ 10 min manual work)

---

**Ready to start? Follow BƯỚC 1 above! 🚀**

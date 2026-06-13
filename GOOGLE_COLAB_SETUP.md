# 🚀 Google Colab Setup for TimesFM Training

## 📦 What You Need to Upload to Google Drive

### Step 1: Create Folder Structure in Google Drive

```
Google Drive/
└── MyDrive/
    └── stockvoli-research/
        ├── configs/
        │   └── config.yaml
        ├── data/
        │   └── processed/
        │       ├── ACB_processed.csv
        │       ├── BCM_processed.csv
        │       ├── BID_processed.csv
        │       └── ... (all 30 stock files)
        ├── src/
        │   └── model_training_fixed.py
        └── colab_timesfm_training.ipynb
```

### Step 2: Files to Upload (Manual Approach)

**Required Files:**
- ✅ `configs/config.yaml` (5.8 KB)
- ✅ `colab_timesfm_training.ipynb` (already created!)
- ✅ `src/model_training_fixed.py`
- ✅ `data/processed/*_processed.csv` (30 files)

**Total Size:** ~16 MB (very manageable!)

---

## 🎯 Quick Start Instructions

### Option A: Manual Upload (Simple but Slow)

1. **Open Google Drive** (drive.google.com)
2. **Create folder:** `stockvoli-research`
3. **Upload these files:**
   - Drag & drop `configs/config.yaml`
   - Drag & drop `colab_timesfm_training.ipynb`
   - Drag & drop `src/model_training_fixed.py`
   - Drag & drop ALL `data/processed/*_processed.csv` files

4. **Open Colab** (colab.google.com)
5. **Upload notebook:** File → Upload notebook → Select `colab_timesfm_training.ipynb`
6. **Run:** Runtime → Run all

### Option B: Automated Setup (Fast - Recommended)

Run this command locally to create a ZIP file ready for upload:

```bash
# Windows Git Bash / WSL
cd D:/bmad-projects/stockvoli-research
bash scripts/prepare_colab_upload.sh
```

This creates `stockvoli-colab-ready.zip` containing everything needed.

Then:
1. Upload `stockvoli-colab-ready.zip` to Google Drive
2. In Colab, run: `!unzip /content/drive/MyDrive/stockvoli-colab-ready.zip -d /content/drive/MyDrive/`

---

## 🔧 What the Notebook Does Automatically

Once you open and run the notebook:

1. **✅ Checks GPU** (T4 16GB)
2. **✅ Installs dependencies** (transformers, peft, torch)
3. **✅ Mounts Google Drive**
4. **✅ Loads VN30 data** (30 stocks)
5. **✅ Trains TimesFM 2.5** with LoRA
6. **✅ Saves checkpoints** to Google Drive
7. **✅ Logs metrics** (QLIKE, R², RMSE, MSE)

---

## 📊 Expected Training Timeline

**On T4 GPU (16GB):**
- Setup & Installation: ~5 minutes
- Data Loading: ~2 minutes
- Training (100 epochs): ~2-3 hours
- Total Time: **~3 hours**

**vs Your Laptop (8GB):**
- Would crash immediately ❌
- If it worked: ~24+ hours ❌

---

## 🎯 After Training Completes

**Results will be saved in Google Drive:**
```
stockvoli-research/
├── models/
│   └── best_model_r2_0.XXX.pt  (Your fine-tuned model!)
└── experiments/
    ├── model_training.log     (Training logs)
    └── training_metrics.json  (Final metrics)
```

**Download your trained model:**
1. Right-click `models/best_model_r2_0.XXX.pt`
2. Download → Save locally
3. Use for inference/prediction

---

## 🚨 Common Issues & Fixes

### Issue: "File not found"
**Fix:** Make sure ALL files are uploaded to Google Drive before running

### Issue: "CUDA out of memory"
**Fix:** This should NOT happen on T4 16GB. If it does, restart runtime.

### Issue: "Module not found"
**Fix:** Make sure you ran "Runtime → Run all" (not individual cells)

### Issue: Training stalls
**Fix:** Check GPU usage with `!nvidia-smi`. Should be ~80-90% utilization.

---

## 💡 Pro Tips

1. **Save your work!** Colab sessions timeout after 90 minutes of inactivity
2. **Monitor progress:** Check the training logs every 30 minutes
3. **Download early:** If R² > 0.5 after 50 epochs, you can stop early
4. **Backup results:** Download model checkpoints immediately after training

---

## 🎯 Success Criteria

Your training is successful when:
- ✅ R² > 0.5 (model explains >50% of variance)
- ✅ QLIKE < 1.0 (low volatility forecast error)
- ✅ RMSE < 0.5 (low prediction error)
- ✅ No CUDA OOM errors (T4 16GB handles it)

---

## 📞 Need Help?

If you run into issues:
1. Check the training logs in `experiments/model_training.log`
2. Check GPU memory with `!nvidia-smi`
3. Verify all files are uploaded correctly
4. Try restarting the Colab runtime

---

**Ready to start training? 🚀**

1. Upload files to Google Drive
2. Open `colab_timesfm_training.ipynb` in Colab
3. Run → Runtime → Run all
4. Wait 3 hours
5. Get your trained model!

Good luck! 🍀

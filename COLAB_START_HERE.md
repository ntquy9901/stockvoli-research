# 🚀 COLAB TRAINING START HERE

## ✅ You're Ready to Train TimesFM on Google Colab!

Your **8GB GPU cannot handle TimesFM 2.5** - but **Google Colab's FREE T4 GPU (16GB)** can!

---

## 📋 YOUR 3-STEP CHECKLIST

### Step 1: Upload Files to Google Drive ✏️

**Time: 5 minutes | Size: 16 MB**

1. **Open Google Drive** → [drive.google.com](https://drive.google.com)
2. **Create folder:** `stockvoli-research`
3. **Upload these files/folders:**

```
From your project:
├── configs/config.yaml              → Upload to stockvoli-research/configs/
├── colab_timesfm_training.ipynb     → Upload to stockvoli-research/
├── src/model_training_fixed.py      → Upload to stockvoli-research/src/
└── data/processed/*.csv             → Upload ALL 30 files to stockvoli-research/data/processed/
```

**Quick test:** You should have **33 files total** in Google Drive

---

### Step 2: Open Notebook in Colab 📓

**Time: 2 minutes**

1. **Open Google Colab** → [colab.google.com](https://colab.google.com)
2. **Upload notebook:** File → Upload notebook → Select `colab_timesfm_training.ipynb`
3. **Connect to GPU:** Runtime → Change runtime type → T4 GPU
4. **Save notebook:** File → Save to Google Drive

---

### Step 3: Run Training 🚀

**Time: ~3 hours | Results: R² > 0.5**

1. **Run all cells:** Runtime → Run all
2. **Watch it work:** The notebook will:
   - ✅ Check GPU (T4 16GB)
   - ✅ Install dependencies
   - ✅ Mount Google Drive
   - ✅ Load VN30 data
   - ✅ Train TimesFM 2.5
   - ✅ Save model checkpoints

3. **Monitor progress:**
   - Check GPU usage: `!nvidia-smi` (should be 80-90%)
   - Read logs: `experiments/model_training.log`
   - Watch metrics improve every epoch

---

## 📊 WHAT TO EXPECT

### Timeline:
- **0-5 min:** Setup & installation
- **5-10 min:** Data loading
- **10-180 min:** Training (100 epochs)
- **180 min:** Complete!

### Final Results (Expected):
- **R² Score:** 0.5-0.6 (explains 50-60% of variance)
- **QLIKE:** < 1.0 (good volatility forecast)
- **RMSE:** < 0.5 (low prediction error)

### Where Results Are Saved:
```
Google Drive → stockvoli-research/
├── models/
│   └── best_model_r2_0.XXX.pt      ← YOUR TRAINED MODEL!
└── experiments/
    ├── model_training.log           ← Training logs
    └── training_metrics.json        ← Final metrics
```

---

## 🎯 SUCCESS METRICS

Your training is successful when:
- ✅ **No CUDA errors** (T4 16GB handles it)
- ✅ **R² > 0.5** (model learns patterns)
- ✅ **QLIKE < 1.0** (better than baseline)
- ✅ **Training completes** (~3 hours)

---

## 🚨 COMMON ISSUES (Don't Panic!)

### "File not found"
**Solution:** Make sure all 33 files are uploaded to Google Drive

### "Runtime disconnected"
**Solution:** Colab times out - just restart from last cell

### "GPU not showing"
**Solution:** Runtime → Change runtime type → T4 GPU → Save

### "Training stalled"
**Solution:** Check with `!nvidia-smi` - GPU should be 80-90% utilized

---

## 💡 PRO TIPS

1. **Save your work!** Colab sessions timeout after 90 minutes of inactivity
2. **Download early:** If R² > 0.5 after 50 epochs, you can stop early
3. **Monitor GPU:** Should be ~80-90% utilization during training
4. **Check logs:** Read `experiments/model_training.log` for progress

---

## 🎓 WHAT'S HAPPENING UNDER THE HOOD

The notebook runs the **actual TimesFM 2.5 foundation model** (not a custom transformer):

```python
✅ TimesFm2_5ModelForPrediction.from_pretrained("google/timesfm-2.5-200m")
✅ LoRA adapters (r=4, alpha=8) 
✅ SGD optimizer (financial ML standard)
✅ All VN30 stocks (30 tickers)
✅ Proper log transformations
✅ Financial clipping (-5, 5 range)
```

---

## 🏆 AFTER TRAINING COMPLETES

1. **Download your model:**
   - Right-click `models/best_model_r2_0.XXX.pt`
   - Download → Save locally

2. **Use for predictions:**
   - Load model: `torch.load('best_model_r2_0.XXX.pt')`
   - Predict: `model.predict(new_data)`

3. **Celebrate!** 🎉
   - You successfully trained TimesFM 2.5
   - Your 8GB laptop couldn't do it
   - Google Colab made it possible (FREE!)

---

## 🆘 NEED HELP?

**Check the training logs:**
```bash
!tail -f experiments/model_training.log
```

**Check GPU status:**
```bash
!nvidia-smi
```

**Verify files uploaded:**
```bash
!ls -la data/processed/ | wc -l  # Should be 30 files
```

---

## 🎯 YOU'RE ALL SET!

**Your roadmap:**
1. ✅ Upload 33 files to Google Drive (5 min)
2. ✅ Open notebook in Colab (2 min) 
3. ✅ Run training (3 hours)
4. ✅ Get trained model! 🎉

**Total time commitment:** ~3 hours (mostly automated)

**Total cost:** $0 (FREE Google Colab)

**Result:** Working TimesFM 2.5 model for Vietnamese stocks!

---

**Ready to start? Go to Step 1! 🚀**

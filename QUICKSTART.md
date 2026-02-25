# 🚀 QUICK START GUIDE

## Brain Tumor Detection Project - Setup in 5 Minutes

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download Dataset
1. Go to [Kaggle Brain MRI Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
2. Download and extract to `data/raw/`
3. Structure should be:
   ```
   data/raw/
   ├── Tumor/
   └── No_Tumor/
   ```

### Step 3: Train Model
```bash
python train.py
```

This will:
- Load and preprocess data
- Train MobileNetV2 model
- Save best model to `models/best_model.h5`
- Generate evaluation plots in `results/plots/`

### Step 4: Test Grad-CAM
```bash
python test_gradcam.py
```

### Step 5: Run Web Application
```bash
cd deployment
python app.py
```

Visit: `http://localhost:5000`

---

## 📊 Using Jupyter Notebooks

### Option 1: EDA
```bash
jupyter notebook notebooks/01_EDA.ipynb
```

### Option 2: Training
```bash
jupyter notebook notebooks/02_Model_Training.ipynb
```

---

## 🎯 Project Structure

```
HILproject/
├── src/                    # Source code
│   ├── data/              # Data loading & preprocessing
│   ├── models/            # Model architectures
│   ├── utils/             # Utilities
│   └── visualization/     # Plots & Grad-CAM
├── deployment/            # Flask web app
├── notebooks/             # Jupyter notebooks
├── data/                  # Dataset
├── models/                # Saved models
├── results/               # Plots & results
├── train.py              # Main training script
└── test_gradcam.py       # Grad-CAM testing
```

---

## 🔧 Configuration

Edit `config.yaml` to customize:
- Model architecture (custom_cnn, mobilenetv2, resnet50)
- Training parameters (epochs, batch size, learning rate)
- Data augmentation settings
- Grad-CAM settings

---

## 📝 Common Issues

### Issue: Model not found
**Solution:** Train the model first: `python train.py`

### Issue: Dataset not found
**Solution:** Download dataset and place in `data/raw/`

### Issue: Out of memory
**Solution:** Reduce batch size in `config.yaml`

---

## 🎓 For Viva/Presentation

### Key Points to Remember:

1. **Why Deep Learning?**
   - Automatic feature extraction
   - Better than traditional ML for images
   - Handles complex patterns

2. **Why Transfer Learning?**
   - Requires less data
   - Faster training
   - Better performance

3. **Why Grad-CAM?**
   - Builds clinical trust
   - Shows what model sees
   - Required for medical AI

4. **Metrics Importance:**
   - Recall > Precision (missing tumor is dangerous)
   - High accuracy not enough
   - Need balanced metrics

---

## 📧 Support

For issues or questions, check:
- README.md (detailed documentation)
- Code comments (inline explanations)
- config.yaml (all settings)

---

**Good luck with your project! 🎉**

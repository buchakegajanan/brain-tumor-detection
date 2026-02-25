# 🧠 Brain Tumor Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-green)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> AI-powered brain tumor detection from MRI scans using deep learning with explainable AI (Grad-CAM) and production-ready web interface.

![Demo](https://img.shields.io/badge/Demo-Live-success)
![Accuracy](https://img.shields.io/badge/Accuracy-98.51%25-brightgreen)
![Recall](https://img.shields.io/badge/Recall-100%25-brightgreen)

## 🎯 Features

- ✅ **Deep Learning Models**: Custom CNN + Transfer Learning (MobileNetV2, ResNet50)
- ✅ **Explainable AI**: Grad-CAM visualization for clinical trust
- ✅ **Web Application**: Flask-based interface with user authentication
- ✅ **Database Integration**: SQLite for user management and prediction history
- ✅ **High Performance**: 98.51% accuracy, 100% recall
- ✅ **Production Ready**: Error handling, logging, deployment configurations

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-detection.git
cd brain-tumor-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download dataset**
- Download from [Kaggle Brain MRI Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- Extract to `data/raw/` folder

4. **Prepare dataset**
```bash
python prepare_dataset.py
```

5. **Train model**
```bash
python train.py
```

6. **Run web application**
```bash
cd deployment
python app.py
```

Visit: `http://localhost:5000`

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 98.51% |
| Recall | 100% |
| Precision | 97.06% |
| F1-Score | 98.51% |
| Specificity | 97% |

## 🏗️ Project Structure

```
brain-tumor-detection/
├── src/                    # Source code
│   ├── data/              # Data processing
│   ├── models/            # Model architectures
│   ├── utils/             # Utilities
│   └── visualization/     # Plots & Grad-CAM
├── deployment/            # Flask web app
├── notebooks/             # Jupyter notebooks
├── data/                  # Dataset (not included)
├── models/                # Saved models (not included)
├── results/               # Results & plots
├── train.py              # Training script
└── README.md             # This file
```

## 🎓 For Academic Use

This project is suitable for:
- Final year engineering projects
- Research papers
- Portfolio showcase
- Learning deep learning and medical AI

## 📝 Documentation

- [Quick Start Guide](QUICKSTART.md)
- [Execution Guide](EXECUTION_GUIDE.md)
- [Complete Guide](docs/COMPLETE_GUIDE.md)
- [Project Summary](PROJECT_SUMMARY.md)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⚠️ Disclaimer

This is an educational project. **NOT intended for clinical use** without proper validation and regulatory approval.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- Dataset: [Kaggle Brain MRI Images](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- Frameworks: TensorFlow, Keras, Flask
- Inspiration: Medical AI research community

---

⭐ Star this repo if you find it helpful!

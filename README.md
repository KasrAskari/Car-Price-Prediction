# 🚗 Car Price Prediction with Deep Learning

A comprehensive exploration of regression-based car price prediction using the `CarPrice_Assignment.csv` dataset. This project implements and compares multiple neural network architectures — including single-layer models, multi-layer networks, an optimized deep model, and a Wide-and-Deep model — to assess performance using **Mean Absolute Error (MAE)** and a custom **Accuracy/R² metric**.

---

## 📦 Overview

The `CarPrice_Assignment.csv` dataset includes features like fuel type, engine size, and horsepower, alongside car prices. This project focuses on predicting prices with deep learning and includes:

-  **Data Preprocessing**: Encoding categorical variables, scaling features, and splitting data (85% train/validation, 15% test).
-  **Single-Layer Models**: Tested with neuron counts of 50, 100, 200, 350, and 500.
-  **Multi-Layer Models**: Built with 100 neurons and layer counts of 1, 2, 5, and 7.
-  **Optimized Deep Model**: A 128-64-32 architecture with Dropout and a custom learning rate.
-  **Wide-and-Deep Model**: Combines wide (shallow) and deep paths for enhanced feature learning.
-  **Evaluation**: Performance measured with MAE and a custom metric (Accuracy if MAE < mean price, otherwise R²).

You can explore the dataset further [here](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction).

---

## 🚀 Quick Start

### ✅ Prerequisites
- Python 3.8+
- `pip` package manager
- Google Colab (optional, for original notebook execution)

### 📥 Installation

1. **Clone the repo:**
   ```bash
   git clone https://github.com/yourusername/car-price-prediction.git
   cd car-price-prediction
   ```

2. **(Optional) Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate        # On Linux/Mac
   venv\Scripts\activate           # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add the dataset:**
   - Place `CarPrice_Assignment.csv` in the project root directory or upload it in Colab.

---

## 📚 Dependencies

Main libraries used:

- `numpy==1.26.4`
- `pandas==2.2.2`
- `tensorflow==2.16.1`
- `matplotlib==3.8.4`
- `scikit-learn==1.4.2`
- `google-colab` (optional, for Colab execution)

(See `requirements.txt` for full list; exclude `google-colab` for local runs.)

---

## 📊 Results

The following table compares performance across models based on the notebook’s output:

```
Results Table:
   Dataset     Single-Layer (200) MAE  Single-Layer Accuracy/R²  Deep MAE  Deep Accuracy/R²  WideDeep MAE  WideDeep Accuracy/R²
0  Train       3000                   77.42%                   1800      86.44%           1900          85.69%
1  Validation  3200                   75.91%                   2000      84.94%           2100          84.19%
2  Test        2672.78                79.87%                   2534.53   80.91%           2763.72       79.18%
```

📌 *Test MAE and Accuracy/R² for Deep and WideDeep models are from the notebook. Others are placeholders; update with your runs.*

---

## 📝 Notes

- Features are normalized using `StandardScaler`.
- Categorical variables (e.g., `fueltype`) are encoded with `LabelEncoder`.
- The optimized deep model includes Dropout (0.2) and a custom Adam optimizer for better generalization.
- Visualizations include a fuel type histogram and MAE vs. neuron/layer plots.
- For local execution, modify the dataset loading from `files.upload()` to a direct file path (e.g., `pd.read_csv('CarPrice_Assignment.csv')`).
- Training time varies; GPU acceleration is recommended (ensure TensorFlow is CUDA-enabled).

---

## 🤝 Contributing

Pull requests are welcome! Feel free to fork the project, open issues, or suggest improvements.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

# 🤖 Auto Model Selector

An end-to-end Machine Learning platform that automates data preprocessing, model training, evaluation, comparison, visualization, and prediction — all through an interactive Flask web interface.

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black.svg)](https://flask.palletsprojects.com/)
[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#-license)
[![GitHub Stars](https://img.shields.io/github/stars/mariamgaber123/Auto-Model-Selector?style=social)](https://github.com/mariamgaber123/Auto-Model-Selector/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/mariamgaber123/Auto-Model-Selector?style=social)](https://github.com/mariamgaber123/Auto-Model-Selector/network/members)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Documentation-purple.svg)](https://deepwiki.com/mariamgaber123/Auto-Model-Selector)

---

## 📖 Overview

Building machine learning models usually involves many repetitive manual steps: data preprocessing, feature engineering, model selection, hyperparameter tuning, evaluation, and prediction. Doing this by hand — for every new dataset, every new problem — is slow, error-prone, and requires solid ML expertise that not every team or student has.

**Auto Model Selector** automates the entire workflow. Users upload a CSV dataset, choose a target column, configure preprocessing options, train multiple ML models, compare their performance side-by-side, visualize results, and generate predictions using the best-performing model — all without writing a single line of code.

### 💡 Why this project matters

- **Lowers the barrier to entry**: students, analysts, and non-technical stakeholders can go from a raw CSV to a working, evaluated ML model through a simple web interface, without writing Python or knowing Scikit-learn.
- **Saves engineering time**: automates the repetitive parts of the ML workflow (cleaning, encoding, scaling, SMOTE, PCA, training, evaluation) that data scientists otherwise repeat manually on nearly every project.
- **Removes guesswork in model selection**: instead of picking one algorithm and hoping it's the right choice, the platform trains and compares multiple models side-by-side on the same metrics, so the "best" model is chosen based on evidence, not assumption.
- **Makes results explainable and visual**: built-in visualizations (correlation heatmaps, distribution plots, box/violin plots, etc.) help users actually understand their data and their model's behavior, instead of just getting a black-box output.
- **Handles real-world data problems out of the box**: imbalanced classes (via SMOTE), high-dimensional data (via PCA), missing values, and mixed categorical/numerical features are all handled automatically — issues that commonly break naive ML pipelines.
- **A strong learning and portfolio project**: it demonstrates a full, production-style ML system — a Flask backend, a modular pipeline architecture, model evaluation, and deployment-ready prediction endpoints — making it a solid reference for anyone learning applied ML engineering.
- **Reusable foundation**: the modular structure (preprocessing, models, evaluation, and visualization as separate subsystems) makes it easy to extend with new models, metrics, or deployment targets later on.

---

## ✨ Features

- 📤 Upload CSV datasets directly from the browser
- 🧹 Automatic data cleaning (missing values, duplicates)
- 🔤 Categorical feature encoding
- 📏 Feature scaling
- 🧬 PCA dimensionality reduction
- ⚖️ SMOTE for handling imbalanced classification datasets
- 🏋️ Train multiple ML models in one run
- 📊 Automatic model comparison and hyperparameter tuning
- 📈 Interactive visualizations (server-rendered plots)
- ✅ Model performance evaluation with standard metrics
- 💾 Download trained models
- 🔮 Real-time prediction interface for new data

---

## 🔄 Project Workflow

```
Upload Dataset
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis (EDA)
      ↓
Encoding
      ↓
Feature Scaling
      ↓
SMOTE (Optional)
      ↓
Feature Selection / PCA
      ↓
Model Training
      ↓
Model Evaluation & Comparison
      ↓
Best Model Selection
      ↓
Prediction
```

---

## 🏗️ Project Architecture

The application is built around a Flask server (`app.py`) that acts as the central orchestrator, tying together the preprocessing pipeline, model training/evaluation, and the visualization dashboard.

```
Frontend (Jinja2 Templates + Bootstrap)
        ↓
Flask Backend (app.py)
        ↓
Preprocessing Pipeline (clean.py, encode.py, pipeline.py, smote.py)
        ↓
Model Training (model_factory.py, train.py)
        ↓
Model Evaluation (evaluate.py)
        ↓
Prediction Engine (predict.py)
        ↓
Visualization Dashboard (plot.py, visualize.html)
```

### How the Flask app works

`app.py` drives the application through four main stages:

1. **Data Ingestion** — uploading and previewing the CSV file.
2. **Configuration** — selecting the target variable and defining model hyperparameters.
3. **Execution** — running the preprocessing and training pipelines (`main.py` for the standard pipeline, `mainSmote.py` for the SMOTE-enabled pipeline).
4. **Analysis** — visualizing data distributions and running real-time predictions.

Because the app is a step-by-step wizard, it keeps track of the session using a few module-level global variables:

| Variable                 | Purpose                                                                   |
| ------------------------ | -------------------------------------------------------------------------- |
| `df_global`               | Holds the active pandas DataFrame after a CSV upload                      |
| `trained_model_global`    | Holds the fitted sklearn pipeline/model                                    |
| `feature_columns_global`  | Feature columns used during training (target excluded)                    |
| `log_transformed_global`  | Flags whether the target went through a `log1p` transformation             |
| `categorical_values`      | Maps categorical columns to their unique values (used to build UI dropdowns) |

Key routes include CSV upload, target selection, model training, `/visualize` (renders server-side Matplotlib plots as Base64 images), and `/predict` (dynamically builds an input form from `feature_columns_global` and returns instant predictions).

📚 Full technical breakdown available on [DeepWiki](https://deepwiki.com/mariamgaber123/Auto-Model-Selector).

---

## 📂 Project Structure

```
Auto-Model-Selector/
├── app.py                 # Flask application & routing
├── main.py                 # Standard ML pipeline
├── mainSmote.py             # SMOTE-enabled ML pipeline
├── preprocessing/           # Cleaning, encoding, scaling, SMOTE
├── models/                  # Model factory, training, evaluation, prediction
├── templates/                # Jinja2 HTML templates
├── static/                    # CSS/JS/static assets
├── reports/                    # Generated evaluation reports
├── visualizations/               # Plotting utilities and outputs
├── datasets/                       # Uploaded/sample datasets
├── utils/                            # Helper utilities
└── requirements.txt
```

---

## 🛠️ Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Plotly
- Joblib
- Bootstrap
- HTML5 / CSS3 / JavaScript

---

## 🤖 Machine Learning Models

| Model                | Classification | Regression |
| --------------------- | :-------------: | :---------: |
| Logistic Regression    | ✅               |             |
| Random Forest           | ✅               | ✅           |
| Decision Tree            | ✅               | ✅           |
| Gradient Boosting         | ✅               | ✅           |
| Linear Regression          |                 | ✅           |

---

## ⚙️ Data Preprocessing

- Missing value handling
- Duplicate removal
- Categorical feature encoding
- Feature scaling
- PCA dimensionality reduction
- SMOTE for imbalanced classification datasets
- Train-test splitting

---

## 📊 Data Visualization

- Correlation Heatmap
- Histograms
- Scatter Plots
- Box Plots
- Violin Plots
- Bar Charts
- Pie Charts
- Bubble Charts
- Pair Plots
- Correlation Matrix

---

## 📈 Model Evaluation

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

---

## 🔮 Prediction

Once the best-performing model is selected, users can enter new feature values through a dynamically generated form and get instant predictions from the app.

---

## 🚀 Installation

```bash
git clone https://github.com/mariamgaber123/Auto-Model-Selector.git

cd Auto-Model-Selector

pip install -r requirements.txt

python app.py
```

The app will start a local Flask server — open the URL shown in your terminal (typically `http://127.0.0.1:5000`) in your browser.

---

## 📖 Usage

1. Upload a CSV dataset.
2. Select the target column.
3. Configure preprocessing options (scaling, PCA, SMOTE, etc.).
4. Train one or more machine learning models.
5. Compare model performance across metrics.
6. Download the best-performing model.
7. Generate predictions using new input data via the `/predict` page.

---

## 📸 Screenshots

> Add screenshots of:
> - Home Page
> - Dataset Upload
> - EDA Dashboard
> - Data Preprocessing
> - Model Training
> - Model Comparison
> - Prediction Interface
> - Visualization Dashboard

---

## 🚀 Future Improvements

- Deep Learning support
- Automated Feature Engineering
- Explainable AI (SHAP & LIME)
- Cloud deployment
- User Authentication
- Experiment Tracking
- Model Versioning

---

## 📚 Documentation

Full architecture, implementation details, and source code documentation are available on **[DeepWiki](https://deepwiki.com/mariamgaber123/Auto-Model-Selector)**, including:

- [Getting Started](https://deepwiki.com/mariamgaber123/Auto-Model-Selector/1.1-getting-started)
- [Flask Web Application](https://deepwiki.com/mariamgaber123/Auto-Model-Selector/2-flask-web-application)
- [ML Pipeline Architecture](https://deepwiki.com/mariamgaber123/Auto-Model-Selector/3-ml-pipeline-architecture)
- [Preprocessing Subsystem](https://deepwiki.com/mariamgaber123/Auto-Model-Selector/4-preprocessing-subsystem)
- [Models Subsystem](https://deepwiki.com/mariamgaber123/Auto-Model-Selector/5-models-subsystem)
- [Visualization Subsystem](https://deepwiki.com/mariamgaber123/Auto-Model-Selector/6-visualization-subsystem)

---

## 👩‍💻 Author

**Mariam Gaber**
Machine Learning Engineer

[GitHub](https://github.com/mariamgaber123)

---



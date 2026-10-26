# 🎓 Student Performance Predictor

A machine learning web application that predicts a student's final Mathematics grade using selected student and study-related features. Built with Python, Scikit-learn, and Streamlit.

**🔗 Live Demo:** [Student Performance Predictor](https://student-performance-predictor-chand.streamlit.app/)

## 📌 Overview

Student Performance Predictor is an interactive machine learning application designed to estimate students' final Mathematics grades based on eight selected features.

The application compares three regression algorithms, evaluates their performance using cross-validation and test-set metrics, and uses the model with the lowest cross-validation Mean Absolute Error (MAE) to generate predictions.

## ✨ Features

- Predicts final Mathematics grades on a 0–20 scale.
- Uses eight selected student-related input features.
- Compares three machine learning regression models.
- Displays model evaluation metrics in a comparison table.
- Provides an interactive and user-friendly Streamlit interface.
- Supports real-time predictions based on user inputs.
- Includes preprocessing for missing values and categorical features.

## 🧠 Input Features

| Feature | Description |
|---|---|
| `age` | Student's age |
| `studytime` | Weekly study-time category |
| `failures` | Number of past class failures, with 3 representing three or more |
| `absences` | Number of school absences |
| `schoolsup` | Extra educational support at school |
| `famsup` | Family educational support |
| `internet` | Internet access at home |
| `paid` | Paid extra Mathematics classes |

**Prediction target:** `G3`, the student's final Mathematics grade in the dataset.

## 🤖 Machine Learning Models

The application trains and compares the following models:

1. **Linear Regression** — establishes a linear baseline for grade prediction.
2. **Random Forest Regressor** — combines multiple decision trees to model potentially nonlinear patterns.
3. **Gradient Boosting Regressor** — builds an ensemble of trees sequentially to improve predictions.

### Model Evaluation

The models are evaluated using:

- **Five-fold cross-validation:** estimates model performance across multiple training folds.
- **Mean Absolute Error (MAE):** measures average absolute prediction error. Lower is better.
- **Root Mean Squared Error (RMSE):** penalizes larger prediction errors more strongly.
- **R² score:** measures how much variation in the target is explained by the model.

The model with the lowest cross-validation MAE is selected for predictions. All models use the same train-test split and cross-validation folds for a fair comparison.

## 🛠️ Technology Stack

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Web Application:** Streamlit
- **Dataset:** UCI Student Performance Dataset — Mathematics records (`student-mat.csv`)
- **Deployment:** Streamlit Community Cloud
- **Version Control:** Git and GitHub

## 📂 Project Structure

```text
student-performance-predictor/
├── app.py
├── student-mat.csv
├── requirements.txt
└── README.md
```

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/student-performance-predictor.git
cd student-performance-predictor
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Start the application

```bash
python -m streamlit run app.py
```

The application will open in your browser. If it does not open automatically, visit `http://localhost:8501`.

## 📦 Dependencies

The project uses the following Python packages:

```text
streamlit
pandas
numpy
scikit-learn
```

These dependencies are listed in `requirements.txt`.

## 🌐 Live Deployment

The application is deployed using Streamlit Community Cloud.

**Try the application:**  
https://student-performance-predictor-chand.streamlit.app/

Users can enter student details, view the model comparison, and obtain an estimated final Mathematics grade.

## ⚠️ Limitations

- The application uses historical Mathematics records from the UCI Student Performance dataset.
- The selected eight features may not capture every factor affecting student performance.
- Predictions may not generalize to other schools, populations, or education systems.
- Observed relationships do not establish cause and effect.
- Predictions are estimates, not guaranteed grades or official academic assessments.

## 🎯 Project Objective

The objective is to demonstrate an end-to-end machine learning workflow, including data preprocessing, regression model comparison, model evaluation, interactive application development, and web deployment.

## 📚 Dataset Acknowledgment

Cortez, P. and Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance.*

The dataset contains student demographic, social, and school-related information collected from Portuguese schools.


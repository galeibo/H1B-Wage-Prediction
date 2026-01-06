# H-1B Visa Prevailing Wage Prediction

## 📌 Project Overview
This project applies advanced machine learning techniques to predict the prevailing wage for H-1B visa applications based on job titles and employer metadata. Using a dataset of over 400,000 records, we explore Natural Language Processing (NLP) methods to extract semantic value from unstructured text and compare dense (SVD) vs. sparse modeling approaches.

**Final Result:** The tuned XGBoost model achieved an **RMSE (log1p) of ~0.208** on the test set.

## 📂 Repository Structure
* **`data/`**: Contains the split datasets (train/val/test).
* **`notebooks/`**:
  * `01_Data_Prep_and_Baselines.ipynb`: Data cleaning, EDA, TF-IDF vectorization, and baseline models (Linear, RF, Stacking).
  * `02_Advanced_Modeling_and_Evaluation.ipynb`: XGBoost implementation (Sparse matrix), Hyperparameter tuning with Optuna, SHAP explainability analysis, and Unsupervised learning (K-Means/DBSCAN).
* **`reports/`**: Contains generated figures (learning curves, SHAP plots) and metric outputs.
* **`requirements.txt`**: List of Python libraries required to run the project.

## 🚀 How to Run
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the notebooks in order (01 then 02).

📊 Key Findings
Sparse vs. Dense: Keeping the raw vocabulary (Sparse TF-IDF) outperformed dimensionality reduction (SVD) for this regression task.

Top Predictors: Employer baseline wage, geographic location (CA, NY), and job seniority keywords ("Principal", "Manager").

Unsupervised: K-Means clustering successfully grouped job titles into professional domains (e.g., Healthcare, Engineering) without labels.

👥 Team

Gal Leibovich

Amit Mirzayev
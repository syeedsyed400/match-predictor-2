# 🏏 Cricket Match Winner Prediction – ML Classification

An end-to-end Machine Learning project that predicts T20 cricket match outcomes (**Team_A** vs **Team_B**) using six classification models, with an interactive **Streamlit** web application for live evaluation.

---

## a. Problem Statement

Given pre-match statistics for a T20 cricket encounter — including team rankings, recent form, head-to-head record, venue characteristics, toss details, and technical indices — the goal is to **predict which team will win** the match. This is a **binary classification** problem with the target labels `Team_A` and `Team_B`.

Accurate match-outcome prediction can assist broadcasters, analysts, and fantasy-league participants in making data-driven decisions before a match begins.

---

## b. Dataset Description

| Property | Detail |
|---|---|
| **Source** | Match_dataset.csv (course-provided) |
| **Instances** | 600 |
| **Features** | 21 (including the target) |
| **Target Variable** | `Winner` (Team_A / Team_B) |
| **Classification Type** | Binary |
| **Missing Values** | None |

### Key Features

| Feature | Description |
|---|---|
| `Team_A_Ranking` / `Team_B_Ranking` | ICC T20 ranking of each team |
| `Team_A_Form` / `Team_B_Form` | Recent performance score (0–100) |
| `HeadToHead_A_Wins` / `HeadToHead_B_Wins` | Historical head-to-head wins |
| `Venue_HomeAdvantage_A` / `Venue_HomeAdvantage_B` | Binary home-ground indicator |
| `Pitch_Type` | Surface category (Flat, Spin-Friendly, etc.) |
| `Avg_T20_Score_Venue` | Average T20 score at the venue |
| `Toss_Winner` / `Toss_Decision` | Toss outcome and elected choice |
| `Team_A_Tech_Index` / `Team_B_Tech_Index` | Composite technical strength index |
| `Match_Total` | Total runs scored in the match |

### Preprocessing Performed

1. **Feature Engineering** – Created difference features (`Ranking_Diff`, `Form_Diff`, `Tech_Diff`, `H2H_Diff`) to capture relative team strength.
2. **Categorical Encoding** – `Pitch_Type` and `Stage` encoded with `LabelEncoder`; `Toss_Winner` converted to binary flag `Team_A_Won_Toss`; `Toss_Decision` mapped to binary `Toss_Bat`.
3. **Scaling** – All features standardised with `StandardScaler`.
4. **Train-Test Split** – 80 / 20 stratified split (`random_state=42`).

---

## c. Models Used

All **6 classification models** are implemented on the same dataset. Each model notebook is located in the [`model/`](model/) folder.

| # | Model | Notebook |
|---|---|---|
| 1 | Logistic Regression | [`model/logistic_regression.ipynb`](model/logistic_regression.ipynb) |
| 2 | Decision Tree Classifier | [`model/decision_tree.ipynb`](model/decision_tree.ipynb) |
| 3 | K-Nearest Neighbors (KNN) | [`model/knn.ipynb`](model/knn.ipynb) |
| 4 | Gaussian Naive Bayes | [`model/naive_bayes.ipynb`](model/naive_bayes.ipynb) |
| 5 | Random Forest (Ensemble) | [`model/random_forest.ipynb`](model/random_forest.ipynb) |
| 6 | XGBoost (Ensemble) | [`model/xgboost_model.ipynb`](model/xgboost_model.ipynb) |

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8500 | 0.9205 | 0.8276 | 0.8571 | 0.8421 | 0.6997 |
| Decision Tree | 0.8000 | 0.7980 | 0.7963 | 0.7679 | 0.7818 | 0.5977 |
| KNN | 0.8000 | 0.8737 | 0.7759 | 0.8036 | 0.7895 | 0.5994 |
| Naive Bayes | 0.8000 | 0.8797 | 0.7759 | 0.8036 | 0.7895 | 0.5994 |
| Random Forest (Ensemble) | 0.8417 | 0.9046 | 0.8246 | 0.8393 | 0.8319 | 0.6824 |
| XGBoost (Ensemble) | 0.8333 | 0.8909 | 0.8214 | 0.8214 | 0.8214 | 0.6652 |

---

### Model Performance Observations

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Achieves the **highest accuracy (85.00 %)** and the **best AUC (0.9205)** among all models. Its strong performance indicates that the relationship between the engineered features and the match outcome is largely linear, making it an excellent baseline. The high recall (0.8571) means it correctly identifies most Team_B wins, and balanced precision/recall indicates reliable predictions for both classes. |
| **Decision Tree** | Records the **lowest AUC (0.7980)**, suggesting that a single tree struggles to generalise beyond the training patterns. While accuracy is 80 %, the precision-recall gap shows slight over-prediction for one class. The model is prone to overfitting on specific feature thresholds, which reduces its discriminative ability on unseen data compared to ensemble approaches. |
| **KNN** | Delivers 80 % accuracy and a solid AUC of 0.8737, confirming that distance-based similarity in the scaled feature space captures useful pattern information. However, it is sensitive to the choice of K and to irrelevant features. Performance can degrade with noisy features since KNN treats all dimensions equally without internal feature selection. |
| **Naive Bayes** | Matches KNN at 80 % accuracy and edges it slightly on AUC (0.8797). Despite the strong independence assumption (which rarely holds for match statistics), Gaussian NB benefits from the well-separated class distributions in the standardised feature space. It is the fastest model to train and ideal for quick baseline comparisons, though it plateaus below ensemble methods. |
| **Random Forest (Ensemble)** | Achieves the **second-best accuracy (84.17 %)** and AUC (0.9046). By aggregating 200 decision trees, it overcomes the single tree's overfitting issue and provides robust predictions. The balanced precision (0.8246) and recall (0.8393) with the highest MCC among non-linear models (0.6824) indicate strong and consistent classification across both classes. |
| **XGBoost (Ensemble)** | Scores 83.33 % accuracy with AUC 0.8909, closely trailing Random Forest. Gradient boosting sequentially corrects errors, giving it excellent calibration. Its precision and recall are perfectly balanced at 0.8214, showing stable predictions. With additional hyperparameter tuning (e.g., learning rate scheduling, regularisation), XGBoost is likely to surpass all other models on this dataset. |

---

## Project Structure

```
ml-match/
│── app.py                        # Streamlit web application
│── requirements.txt              # Python dependencies
│── README.md                     # Project documentation
│── Match_dataset.csv             # Dataset
│── model/                        # Model notebooks (.ipynb)
│   ├── model_comparison.ipynb
│   ├── logistic_regression.ipynb
│   ├── decision_tree.ipynb
│   ├── knn.ipynb
│   ├── naive_bayes.ipynb
│   ├── random_forest.ipynb
│   └── xgboost_model.ipynb
```

---

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/ml-match.git
cd ml-match

# 2. Create a virtual environment & install dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. (Optional) Re-train all models – run model/model_comparison.ipynb

# 4. Launch the Streamlit app
streamlit run app.py
```

---

## Streamlit App Features

| Feature | Description |
|---|---|
| **CSV Upload** | Upload your own test CSV (same schema) for evaluation |
| **Model Selection** | Multi-select dropdown to choose one or more models |
| **Evaluation Metrics** | Accuracy, AUC, Precision, Recall, F1, MCC displayed in a comparison table |
| **Confusion Matrix** | Heatmap per selected model |
| **Classification Report** | Per-class precision, recall, F1-score |
| **Bar Chart Comparison** | Side-by-side metric comparison across models |

---

## Technologies Used

- Python 3.12
- scikit-learn
- XGBoost
- Streamlit
- pandas / NumPy
- matplotlib / seaborn

---

## License

This project is for academic / educational purposes.

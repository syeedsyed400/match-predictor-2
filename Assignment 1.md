# 1. Overview  
  
In this assignment, you are required to:  
  
- Implement multiple classification models    
- Build an interactive Streamlit web application to demonstrate your models    
- Deploy the app on Streamlit Community Cloud (FREE)    
- Share clickable links for evaluation    
  
You will learn real-world end-to-end ML deployment workflow: modeling, evaluation, UI design, and deployment.  
  
  
---  
  
# 2. Mandatory Submission Links  
  
Each submission must be a **single PDF file** with the following (maintain the order):  
  
## 1. GitHub Repository Link containing  
- Complete source code    
- `requirements.txt`    
- A clear `README.md`    
  
## 2. Live Streamlit App Link  
- Deployed using Streamlit Community Cloud    
- Must open an interactive frontend when clicked    
  
 

## 4. README Content in Submission  
  
The GitHub `README.md` content (details mentioned in **Section 3 – Step 5**) should also be part of the **submitted PDF file**.  
  

  
---  
  
# 3. Assignment Details  
  
## Step 1: Dataset  

Use Match_dataset.csv classification dataset
  
 
  
---  
  
## Step 2: Machine Learning Classification Models and Evaluation Metrics  
  
Implement the following classification models using the dataset chosen above.    
**All 6 ML models must be implemented on the same dataset.**  
  
1. Logistic Regression    
2. Decision Tree Classifier    
3. K-Nearest Neighbor (KNN) Classifier    
4. Naive Bayes Classifier (Gaussian or Multinomial)    
5. Ensemble Model – Random Forest    
6. Ensemble Model – XGBoost    
  
### Evaluation Metrics  
  
For **each model**, calculate the following evaluation metrics:  
  
1. Accuracy    
2. AUC Score    
3. Precision    
4. Recall    
5. F1 Score    

6. Matthews Correlation Coefficient (MCC Score)  
  
  
**[1 mark]**  
  
---  
  
## Step 3: Prepare Your GitHub Repository  
  
Your repository must contain the following structure:  
  
```text  
project-folder/  
│-- app.py (or streamlit_app.py)  
│-- requirements.txt  
│-- README.md  
│-- model/ (saved model files for all implemented models – *.py or *.ipynb)  

```

## Step 4: Create `requirements.txt`  
  
### Example:  
  
```text  
streamlit  
scikit-learn  
numpy  
pandas  
matplotlib  
seaborn  
```

> **Note:**    
> Missing dependencies are the **#1 cause of deployment failure**.  
  
---  
  
## Step 5: `README.md` Structure  
  
The `README.md` must follow the structure below.    
This **README content should also be part of the submitted PDF file**.    
Follow the required structure carefully.  
  
### a. Problem Statement  
  
Clearly describe the machine learning classification problem you are solving.  
  
---  
  
### b. Dataset Description **[1 mark]**  
  
Provide details about the dataset used, including:  
- Dataset source (Kaggle)  
- Number of instances  
- Number of features  
- Type of classification (binary / multi-class)  
- Any preprocessing performed  
  
---  
  
### c. Models Used **[6 marks]**  
  
- Implement **all 6 models**  
- **1 mark for all the metrics for each model**  
- Create a **comparison table** with evaluation metrics calculated for all models  
  
#### Model Comparison Table  
  
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |  
|--------------|----------|-----|-----------|--------|----|-----|  
| Logistic Regression |  |  |  |  |  |  |  
| Decision Tree |  |  |  |  |  |  |  
| KNN |  |  |  |  |  |  |  
| Naive Bayes |  |  |  |  |  |  |  
| Random Forest (Ensemble) |  |  |  |  |  |  |  
| XGBoost (Ensemble) |  |  |  |  |  |  |  
  

- Add your observations on the performance of each model on the chosen dataset. **[3 marks]**  
  
### Model Performance Observations  
  
| ML Model Name | Observation about Model Performance |  
|--------------|-------------------------------------|  
| Logistic Regression |  |  
| Decision Tree |  |  
| KNN |  |  
| Naive Bayes |  |  
| Random Forest (Ensemble) |  |  
| XGBoost (Ensemble) |  |  

## Step 6: Deploy on Streamlit Community Cloud  
  
1. Go to https://streamlit.io/cloud    
2. Sign in using your GitHub account    
3. Click **New App**    
4. Select your repository    
5. Choose the branch (usually `main`)    
6. Select `app.py`    
7. Click **Deploy**  

Within a few minutes, your app will be live.  
  
---  
  
## Streamlit App – Mandatory Features  
  
Your Streamlit app must include **at least** the following features:  
  
a. **Dataset upload option (CSV)**    
   *As Streamlit free tier has limited capacity, upload only test data.*    
   **[1 mark]**  
  
b. **Model selection dropdown (multiple models)**    
   **[1 mark]**  
  
c. **Display of evaluation metrics**    
   **[1 mark]**  
  
d. **Confusion matrix or classification report**    
   **[1 mark]**  
  
---  
  
## 5. Anti‑Plagiarism & Academic Integrity Guidelines  
  
To ensure originality, the following checks will be performed.    
**Any plagiarism found will result in ZERO (0) marks.**  
  
### Code‑Level Checks  
- GitHub commit history will be reviewed  
- Identical repository structure and variable names may be flagged  
  
### UI‑Level Checks  
- Copy‑paste Streamlit templates without customization may be penalized  
  
### Model‑Level Checks  
- Same dataset + same model + same outputs across students will be investigated  
  
> **Note:**    
> Using AI tools is allowed **only for learning support**, not for direct copy‑paste submissions.  
  
---  
  
## 8. Final Submission Checklist (Before You Submit)  
  
- [ ] GitHub repository link works    
- [ ] Streamlit app link opens correctly    
- [ ] App loads without errors    
- [ ] All required features are implemented    
- [ ] `README.md` updated and included in the submitted PDF  

## Assignment Evaluation & Submission Details  
  
The assignment is for **15 marks**:  
  
- **Model implementation and uploading on GitHub:** **10 marks**  
- **Streamlit app development:** **4 marks**  
  
---  
  
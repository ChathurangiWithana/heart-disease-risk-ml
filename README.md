# Heart Disease Risk ML
Machine learning analysis of the UCI Heart Disease dataset by Kaggle using classification, regression, and clustering methods.  
The project explores supervised and unsupervised learning approaches to predict disease status, estimate continuous outcomes, and identify patient risk profiles.

## 📌 Project Overview
Cardiovascular disease is a leading cause of death worldwide. Early detection and risk profiling can support prevention and treatment.  
This project applies multiple machine learning approaches to the UCI Heart Disease dataset:

- **Classification** → Predict whether a patient has heart disease (`condition`)  
- **Regression** → Predict maximum heart rate achieved (`thalach`)  
- **Clustering** → Group patients into low- and high-risk profiles

## 📂 Dataset
- Source: [UCI Machine Learning Repository – Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)  
- Number of records: ~300  
- Number of features: 13 clinical attributes + target (`condition`)  

**Key variables:**
- `age`: Age in years  
- `sex`: (1 = male, 0 = female)  
- `cp`: Chest pain type (0 = typical angina, 1 = atypical, 2 = non-anginal, 3 = asymptomatic)  
- `trestbps`: Resting blood pressure (mm Hg)  
- `chol`: Serum cholesterol (mg/dl)  
- `fbs`: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)  
- `restecg`: Resting ECG results (0 = normal, 1 = abnormal, 2 = LV hypertrophy)  
- `thalach`: Maximum heart rate achieved  
- `exang`: Exercise-induced angina (1 = yes, 0 = no)  
- `oldpeak`: ST depression induced by exercise  
- `slope`: Slope of ST segment (0 = upsloping, 1 = flat, 2 = downsloping)  
- `ca`: Number of major vessels (0–3) colored by fluoroscopy  
- `thal`: Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)  
- `condition`: (0 = no disease, 1 = disease)


## ⚙️ Methods
## 🔸 Classification: Heart Disease Prediction

We applied several machine learning models to predict the presence of heart disease (`condition`) from patient features.

### With and without feature scaling
- **No feature scaling**: Models like Logistic Regression and Random Forest already performed well without scaling.
- **With feature scaling**: Scaling improved performance for distance-based models such as KNN and SVM, but had little effect on tree-based models.

### Results (test set)
- **Logistic Regression**: ~83% accuracy (best performing, balanced precision & recall)  
- **Random Forest**: ~77% accuracy, good recall but slightly lower overall accuracy  
- **SVM**: ~67% accuracy, improved with scaling but still weaker  
- **KNN**: ~50% accuracy without scaling, improved with scaling but still underperformed  
- **Decision Tree**: ~73% accuracy, risk of overfitting  

### Interpretation
- Logistic Regression provided the most reliable and interpretable performance.  
- Feature scaling had a clear impact on models sensitive to distance metrics (KNN, SVM).  
- Tree-based models (Random Forest, Decision Tree) were less affected by scaling.  

**Conclusion:** Logistic Regression was the best model for this classification task, while scaling was critical for distance-based methods.


### 🔸 Regression
- Target: Maximum heart rate achieved (`thalach`)  
- Models: Linear Regression, Decision Tree Regression, Random Forest Regression  
- Metrics: Mean Squared Error (MSE), R² score  
- Key finding: Linear Regression with selected features explained ~42% variance in HR  

### 🔸 Clustering
- Target: Can we group patients into **low-risk** and **high-risk** profiles without using the disease label?
- Features: age, chest pain type, cholesterol, oldpeak, resting BP, thalach, slope  
- Method: KMeans (k=2), PCA visualization  
- Result: Two clusters emerged:  
  - **Cluster 0** → Mostly low-risk patients (25% disease prevalence)  
  - **Cluster 1** → Mostly high-risk patients (77% disease prevalence)  

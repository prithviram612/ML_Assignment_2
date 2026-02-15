**1. Problem Statement**

The objective of this project is to implement and compare six classification algorithms on a real-world dataset. The task is to predict whether a bank client will subscribe to a term deposit based on demographic and campaign-related features.

The project includes:

Implementation of six machine learning models

Evaluation using multiple performance metrics

Comparison of model performance

Deployment of a Streamlit-based interactive dashboard

**2. Dataset Description**

The objective of this project is to implement and compare six classification algorithms on a real-world dataset. The task is to predict whether a bank client will subscribe to a term deposit based on demographic and campaign-related features.

The project includes:

Implementation of six machine learning models

Evaluation using multiple performance metrics

Comparison of model performance

Deployment of a Streamlit-based interactive dashboard

**3. Models Used**
The following six classification algorithms were implemented:

1. Logistic Regression

2. Decision Tree

3. K-Nearest Neighbors (KNN)

4. Naive Bayes

5. Random Forest

6. XGBoost

                 Model  Accuracy       AUC  Precision    Recall        F1       MCC
   
0  Logistic_Regression  0.887869  0.870001   0.596977  0.217232  0.318548  0.313371

1        Decision_Tree  0.894393  0.861473   0.591153  0.404216  0.480131  0.433149

2                  KNN  0.891187  0.826034   0.585875  0.334555  0.425904  0.388523

3          Naive_Bayes  0.824837  0.809394   0.342693  0.492209  0.404063  0.312110

4        Random_Forest  0.897822  0.910702   0.689342  0.278643  0.396867  0.395332

5              XGBoost  0.908437  0.931857   0.667090  0.481210  0.559105  0.517977

**4. Observations**

Among the six implemented models, XGBoost demonstrated the strongest overall performance on this dataset.

Reasons:

It effectively captures nonlinear relationships.

It handles complex feature interactions.

It performs well on structured tabular data.

It achieved the highest balance between precision and recall.

Random Forest also performed strongly and showed stable results, but XGBoost achieved slightly better overall performance metrics.

Other observations:

  a. Logistic Regression performed well for linear separation.

  b. Decision Tree showed slight overfitting.

  c. KNN was sensitive to scaling.

  d. Naive Bayes performed decently despite independence assumption.

1. Problem Statement

Predict whether a bank client will subscribe to a term deposit.

2. Dataset Description

The dataset used for this project is the Bank Marketing Dataset obtained from the UCI Machine Learning Repository. This dataset contains information related to direct marketing campaigns conducted by a Portuguese banking institution. The objective of the dataset is to predict whether a client will subscribe to a term deposit based on various personal and campaign-related attributes.

The dataset consists of 45,211 instances and 16 input features, along with one target variable. The target variable, labeled as "y", indicates whether the client subscribed to a term deposit ("yes") or not ("no"). Therefore, this is a binary classification problem.

The dataset includes both numerical and categorical features. Numerical features include attributes such as age, balance, duration, campaign, pdays, and previous. Categorical features include job type, marital status, education level, contact type, month of contact, and previous campaign outcome.

Before model training, categorical variables were encoded into numerical form, and feature scaling was applied to ensure uniformity across models that are sensitive to feature magnitude.

This dataset satisfies the assignment requirements as it contains more than 500 instances and more than 12 features. It is suitable for evaluating multiple classification algorithms due to its real-world complexity and class imbalance characteristics.

3. Models Used

                 Model  Accuracy       AUC  Precision    Recall  F1 Score       MCC
0  Logistic Regression  0.887869  0.870001   0.596977  0.217232  0.318548  0.313371
1        Decision Tree  0.872387  0.706006   0.472000  0.486709  0.479242  0.406608
2                  KNN  0.891187  0.826034   0.585875  0.334555  0.425904  0.388523
3          Naive Bayes  0.824837  0.809394   0.342693  0.492209  0.404063  0.312110
4        Random Forest  0.900807  0.922798   0.637394  0.412466  0.500835  0.461677
5              XGBoost  0.908106  0.928546   0.656250  0.500458  0.567863  0.523443

4. Observations

  a. Logistic Regression performed well for linear separation.

  b. Decision Tree showed slight overfitting.

  c. KNN was sensitive to scaling.

  d. Naive Bayes performed decently despite independence assumption.

  e. Random Forest improved stability.

  f. XGBoost achieved highest AUC score.

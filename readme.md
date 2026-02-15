1. Problem Statement

Predict whether a bank client will subscribe to a term deposit.

2. Dataset Description

16 input features

4000+ instances

Binary target variable

3. Models Used

                 Model  Accuracy       AUC  Precision    Recall  F1 Score       MCC
0  Logistic Regression  0.887869  0.870001   0.596977  0.217232  0.318548  0.313371
1        Decision Tree  0.872830  0.700721   0.473010  0.473877  0.473443  0.401126
2                  KNN  0.891187  0.826034   0.585875  0.334555  0.425904  0.388523
3          Naive Bayes  0.824837  0.809394   0.342693  0.492209  0.404063  0.312110
4        Random Forest  0.902798  0.923533   0.649718  0.421632  0.511395  0.473413
5              XGBoost  0.908106  0.928546   0.656250  0.500458  0.567863  0.523443

4. Observations

Write:

Logistic Regression performed well for linear separation.

Decision Tree showed slight overfitting.

KNN was sensitive to scaling.

Naive Bayes performed decently despite independence assumption.

Random Forest improved stability.

XGBoost achieved highest AUC score.
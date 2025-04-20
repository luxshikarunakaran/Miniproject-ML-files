import numpy as np # linear algebra
import pandas as pd

df = pd.read_csv("survey_lung_cancer.csv")
df.head()

df['GENDER'] = df['GENDER'].replace({'M':0,'F':1})
df.head()

df['LUNG_CANCER'] = df['LUNG_CANCER'].replace({'YES':1,'NO':0})
df.head()

x = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']

#---------------------------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)

print(f"x:train{x_train.shape} x_test:{x_test.shape} y_train:{y_train.shape} y_test:{y_test.shape}")

np.random.seed(32)

# GaussianNB
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train, y_train)

print(model.score(x_train,y_train))
print(model.score(x_test, y_test))

y_pred = model.predict(x_test)

from sklearn.metrics import confusion_matrix , classification_report

cnfm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(cnfm)
print(report)

from sklearn.model_selection import StratifiedKFold , LeaveOneOut , KFold
from sklearn.model_selection import cross_val_score

# stratified k-fold cross validation
skf = StratifiedKFold(n_splits=5)
skfold_score = cross_val_score(model,x,y,cv=skf)
skfold_score_mean = skfold_score.mean()
print(skfold_score)
print("The average score for skfold: ",skfold_score_mean)

#leave out method
loovc = LeaveOneOut()
loocv_score = cross_val_score(model,x,y,cv=loovc)
loocv_score_mean = loocv_score.mean()
print(loocv_score)
print(" The average accuracy for loovc is : ",loocv_score_mean)

kf = KFold(n_splits=5)
k_fold_score = cross_val_score(model,x,y,cv=kf)
k_fold_score_mean = k_fold_score.mean()
print(k_fold_score)
print("The k-fold score mean : ",k_fold_score_mean)

#----------------------------------------------------------------------------
#SVM

from sklearn.svm import SVC
model2 = SVC()
model2.fit(x_train, y_train)

y_pred2 = model2.predict(x_test)

print(model2.score(x_train,y_train))
print(model2.score(x_test, y_test))

from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred2)
print(report)

# stratified k-fold cross validation

skf = StratifiedKFold(n_splits=5)
skfold_score = cross_val_score(model2,x,y,cv=skf)
skfold_score_mean = skfold_score.mean()
print(skfold_score)
print(skfold_score_mean)

loocv = LeaveOneOut()
loocv_score = cross_val_score(model2,x,y,cv=loocv)
loocv_score_mean = loocv_score.mean()
print("The accuracy of Leave one out method is , " , loocv_score)
print("the average accuracy is : ", loocv_score_mean)

kf = KFold(n_splits=5)
score = cross_val_score(model2,x,y,cv=kf)
avg_score = score.mean()
print(score)
print("the average score is : ", avg_score)

#-------------------------------------------------------------------------------------
#Logisitc regression

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut, cross_val_score
from sklearn.metrics import classification_report

# Initialize Logistic Regression model
model_lr = LogisticRegression()
model_lr.fit(x_train, y_train)


# Predictions
y_pred_lr = model_lr.predict(x_test)

# Model evaluation
print(model_lr.score(x_train, y_train))  # Training accuracy
print(model_lr.score(x_test, y_test))    # Testing accuracy

# Classification report
report_lr = classification_report(y_test, y_pred_lr)
print(report_lr)


# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5)
skfold_score_lr = cross_val_score(model_lr, x, y, cv=skf)
skfold_score_mean_lr = skfold_score_lr.mean()
print(skfold_score_lr)
print(skfold_score_mean_lr)


# Leave-One-Out Cross-Validation
loocv = LeaveOneOut()
loocv_score_lr = cross_val_score(model_lr, x, y, cv=loocv)
loocv_score_mean_lr = loocv_score_lr.mean()
print("The accuracy of Leave-One-Out method is:", loocv_score_lr)
print("The average accuracy is:", loocv_score_mean_lr)


# K-Fold Cross-Validation
kf = KFold(n_splits=5)
kf_score_lr = cross_val_score(model_lr, x, y, cv=kf)
kf_avg_score_lr = kf_score_lr.mean()
print(kf_score_lr)
print("The average score is:", kf_avg_score_lr)

#---------------------------------------------------------------------------------------

#Decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut, cross_val_score
from sklearn.metrics import classification_report

# Initialize Decision Tree model
model_dt = DecisionTreeClassifier()
model_dt.fit(x_train, y_train)

# Predictions
y_pred_dt = model_dt.predict(x_test)

# Model evaluation
print(model_dt.score(x_train, y_train))  # Training accuracy
print(model_dt.score(x_test, y_test))

# Classification report
report_dt = classification_report(y_test, y_pred_dt)
print(report_dt)

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5)
skfold_score_dt = cross_val_score(model_dt, x, y, cv=skf)
skfold_score_mean_dt = skfold_score_dt.mean()
print(skfold_score_dt)
print(skfold_score_mean_dt)

# Leave-One-Out Cross-Validation
loocv = LeaveOneOut()
loocv_score_dt = cross_val_score(model_dt, x, y, cv=loocv)
loocv_score_mean_dt = loocv_score_dt.mean()
print("The accuracy of Leave-One-Out method is:", loocv_score_dt)
print("The average accuracy is:", loocv_score_mean_dt)

# K-Fold Cross-Validation
kf = KFold(n_splits=5)
kf_score_dt = cross_val_score(model_dt, x, y, cv=kf)
kf_avg_score_dt = kf_score_dt.mean()
print(kf_score_dt)
print("The average score is:", kf_avg_score_dt)

#---------------------------------------------------------------------------------------

# Random forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut, cross_val_score
from sklearn.metrics import classification_report

# Initialize Random Forest model
model_rf = RandomForestClassifier()
model_rf.fit(x_train, y_train)

# Predictions
y_pred_rf = model_rf.predict(x_test)

# Model evaluation
print(model_rf.score(x_train, y_train))  # Training accuracy
print(model_rf.score(x_test, y_test))    # Testing accuracy

# Classification report
report_rf = classification_report(y_test, y_pred_rf)
print(report_rf)

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5)
skfold_score_rf = cross_val_score(model_rf, x, y, cv=skf)
skfold_score_mean_rf = skfold_score_rf.mean()
print(skfold_score_rf)
print(skfold_score_mean_rf)

# Leave-One-Out Cross-Validation
loocv = LeaveOneOut()
loocv_score_rf = cross_val_score(model_rf, x, y, cv=loocv)
loocv_score_mean_rf = loocv_score_rf.mean()
print("The accuracy of Leave-One-Out method is:", loocv_score_rf)
print("The average accuracy is:", loocv_score_mean_rf)

# K-Fold Cross-Validation
kf = KFold(n_splits=5)
kf_score_rf = cross_val_score(model_rf, x, y, cv=kf)
kf_avg_score_rf = kf_score_rf.mean()
print(kf_score_rf)
print("The average score is:", kf_avg_score_rf)

#-------------------------------------------------------------------------------------

# gradient boosting

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut, cross_val_score
from sklearn.metrics import classification_report

# Initialize Gradient Boosting model
model_gb = GradientBoostingClassifier()
model_gb.fit(x_train, y_train)

# Predictions
y_pred_gb = model_gb.predict(x_test)

# Model evaluation
print(model_gb.score(x_train, y_train))  # Training accuracy
print(model_gb.score(x_test, y_test))    # Testing accuracy

# Classification report
report_gb = classification_report(y_test, y_pred_gb)
print(report_gb)

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5)
skfold_score_gb = cross_val_score(model_gb, x, y, cv=skf)
skfold_score_mean_gb = skfold_score_gb.mean()
print(skfold_score_gb)
print(skfold_score_mean_gb)

# Leave-One-Out Cross-Validation
loocv = LeaveOneOut()
loocv_score_gb = cross_val_score(model_gb, x, y, cv=loocv)
loocv_score_mean_gb = loocv_score_gb.mean()
print("The accuracy of Leave-One-Out method is:", loocv_score_gb)
print("The average accuracy is:", loocv_score_mean_gb)

# K-Fold Cross-Validation
kf = KFold(n_splits=5)
kf_score_gb = cross_val_score(model_gb, x, y, cv=kf)
kf_avg_score_gb = kf_score_gb.mean()
print(kf_score_gb)
print("The average score is:", kf_avg_score_gb)

# ---------------------------------------------------------------------

# XGBoost

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut, cross_val_score
from sklearn.metrics import classification_report

# Initialize XGBoost model
model_xgb = XGBClassifier()
model_xgb.fit(x_train, y_train)

# Predictions
y_pred_xgb = model_xgb.predict(x_test)

# Model evaluation
print(model_xgb.score(x_train, y_train))  # Training accuracy
print(model_xgb.score(x_test, y_test))    # Testing accuracy

# Classification report
report_xgb = classification_report(y_test, y_pred_xgb)
print(report_xgb)

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5)
skfold_score_xgb = cross_val_score(model_xgb, x, y, cv=skf)
skfold_score_mean_xgb = skfold_score_xgb.mean()
print(skfold_score_xgb)
print(skfold_score_mean_xgb)

# Leave-One-Out Cross-Validation
loocv = LeaveOneOut()
loocv_score_xgb = cross_val_score(model_xgb, x, y, cv=loocv)
loocv_score_mean_xgb = loocv_score_xgb.mean()
print("The accuracy of Leave-One-Out method is:", loocv_score_xgb)
print("The average accuracy is:", loocv_score_mean_xgb)

# K-Fold Cross-Validation
kf = KFold(n_splits=5)
kf_score_xgb = cross_val_score(model_xgb, x, y, cv=kf)
kf_avg_score_xgb = kf_score_xgb.mean()
print(kf_score_xgb)
print("The average score is:", kf_avg_score_xgb)

# -----------------------------------------------------------------------

# linear regression

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# Initialize Linear Regression model
model_lre = LinearRegression()
model_lre.fit(x_train, y_train)

# Predictions
y_pred_lre = model_lre.predict(x_test)

# Model evaluation
print("Training R^2 Score:", model_lre.score(x_train, y_train))  # Training R² score
print("Testing R^2 Score:", model_lre.score(x_test, y_test))    # Testing R² score

# Mean Squared Error and R² Score
mse = mean_squared_error(y_test, y_pred_lre)
r2 = r2_score(y_test, y_pred_lre)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# K-Fold Cross-Validation
kf = KFold(n_splits=5)
kf_score_lre = cross_val_score(model_lre, x, y, cv=kf, scoring='r2')
kf_avg_score_lre = kf_score_lre.mean()
print(kf_score_lre)
print("The average R^2 score is:", kf_avg_score_lr)

# StratifiedKFold
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Convert continuous target values into bins (e.g., 5 quantiles)
num_bins = 5
y_binned = np.digitize(y, bins=np.linspace(min(y), max(y), num_bins))

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skfold_score_lre = cross_val_score(model_lr, x, y, cv=skf, scoring='r2')
skfold_score_mean_lre = skfold_score_lre.mean()
print("Stratified K-Fold R^2 Scores:", skfold_score_lre)
print("Average R^2 Score from Stratified K-Fold:", skfold_score_mean_lre)







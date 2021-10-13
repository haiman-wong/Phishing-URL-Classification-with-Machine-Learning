#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns  
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics, svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
import string
from nltk.corpus import stopwords
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import accuracy_score,roc_curve,auc, confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[66]:


# import dataset

df = pd.read_csv('combined_dataset.csv')
df = df.drop('domain', axis=1)
df1=df
df4=df
df5=df
df11=df


# In[67]:


for col in df1.columns:
    unique_value_list = df[col].unique()
    if len(unique_value_list) > 10:
        print(f'{col} has {df[col].nunique()} unique values')
    else:
        print(f'{col} contains:\t\t\t{unique_value_list}')


# In[68]:


col_names = ['ranking', 'isIp', 'valid', 'activeDuration', 'urlLen', 'is@',
       'isredirect', 'haveDash', 'domainLen', 'nosOfSubdomain', 'label']

feature_col = ['ranking', 'isIp', 'valid', 'activeDuration', 'urlLen', 'is@',
       'isredirect', 'haveDash', 'domainLen', 'nosOfSubdomain']


# In[69]:


sns.countplot(x="label",data=df)


# In[70]:


# Confusion matrix chart function 


def plot_confusion_matrix(test_Y, predict_y):
 C = confusion_matrix(test_Y, predict_y)
 A =(((C.T)/(C.sum(axis=1))).T)
 B =(C/C.sum(axis=0))
 plt.figure(figsize=(20,4))
 labels = [1,2]
 cmap=sns.light_palette("orange")
 plt.subplot(1, 3, 1)
 sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
 plt.xlabel('Predicted Class')
 plt.ylabel('Original Class')
 plt.title("Confusion matrix")
 #plt.subplot(1, 3, 2)
 '''sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
 plt.xlabel('Predicted Class')
 plt.ylabel('Original Class')
 plt.title("Precision matrix")
 plt.subplot(1, 3, 3)
 sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
 plt.xlabel('Predicted Class')
 plt.ylabel('Original Class')
 plt.title("Recall matrix")'''
 plt.show()


# # DECISION TREES

# In[71]:


# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('label', axis=1), df['label'],
    test_size=0.25)


# In[72]:


classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
predicted_dt = classifier.predict(X_test)

X = col_names
y = df['label'].values


print("Accuracy: %.3f" % metrics.accuracy_score(y_test, predicted_dt))


# In[73]:


#Decision Tree Variation 1
lab_clf1 = DecisionTreeClassifier(criterion='entropy', max_depth=5, max_leaf_nodes=5)
lab_clf1.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(lab_clf1.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(lab_clf1.score(X_test, y_test)))


# In[74]:


#2 Decision Tree Variation 2
lab_clf2 = DecisionTreeClassifier(criterion='entropy', max_depth=7, max_leaf_nodes=7)
lab_clf2.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(lab_clf2.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(lab_clf2.score(X_test, y_test)))


# In[75]:


#Decision Tree Variation 3
lab_clf3 = DecisionTreeClassifier(criterion='entropy', max_depth=15, min_samples_leaf=15)
lab_clf3.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(lab_clf3.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(lab_clf3.score(X_test, y_test)))


# In[76]:


#Decision Tree Variation 4
lab_clf4 = DecisionTreeClassifier(criterion='gini', max_depth=300, max_leaf_nodes=300)
lab_clf4.fit(X_train, y_train)
print("Accuracy on training set: {:.5f}".format(lab_clf4.score(X_train, y_train)))
print("Accuracy on test set: {:.5f}".format(lab_clf4.score(X_test, y_test)))


# In[77]:


#Decision Tree Variation 5
lab_clf5 = DecisionTreeClassifier(criterion='gini', max_depth=10000, max_leaf_nodes=10000)
lab_clf5.fit(X_train, y_train)
print("Accuracy on training set: {:.5f}".format(lab_clf5.score(X_train, y_train)))
print("Accuracy on test set: {:.5f}".format(lab_clf5.score(X_test, y_test)))


# In[78]:


#Decision Tree Variation 6
lab_clf6 = DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=600, max_leaf_nodes=600)
lab_clf6.fit(X_train, y_train)
print("Accuracy on training set: {:.5f}".format(lab_clf6.score(X_train, y_train)))
print("Accuracy on test set: {:.5f}".format(lab_clf6.score(X_test, y_test)))


# In[79]:


#Define a dictionary containing all hyperparameter values to try:
dt = DecisionTreeClassifier()
hyperparam_grid = {
    "criterion":['gini', 'entropy'],
    "max_depth":range(1,10),
    "min_samples_split":(2,5,10),
    "min_samples_leaf": range(1,5),
}


# In[80]:


classifier = GridSearchCV(dt, hyperparam_grid)
classifier.fit(X_train, y_train)
predicted_gs_classifier = classifier.predict(X_test)
print("Accuracy: %.3f" % metrics.accuracy_score(y_test, predicted_gs_classifier ))
print('Best Criterion: %s' % classifier.best_estimator_.criterion)
print('Best Max_Depth: %s' % classifier.best_estimator_.max_depth)
print('Best Min_Samples_Split: %s' % classifier.best_estimator_.min_samples_split)
print('Best Min_Samples_Leaf: %s' % classifier.best_estimator_.min_samples_leaf)


# In[81]:



dt.fit(X_train, y_train)
predictions = dt.predict(X_test)
print('Decision Tree Accuracy: ', dt.score(X_test, y_test))
print('Decision Tree AUC: ', metrics.roc_auc_score(y_test, dt.predict_proba(X_test)[:,1]))


# In[82]:


fpr, tpr, thresholds = metrics.roc_curve(y_test, dt.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[83]:


# save confusion matrix and slice into four pieces
c_matrix = confusion_matrix(y_test, predictions)
TP = c_matrix[1, 1]
TN = c_matrix[0, 0]
FP = c_matrix[0, 1]
FN = c_matrix[1, 0]
print(confusion_matrix(y_test, predictions))
print("False Positive Rate: ", FP / float(FP + TN))



confmatrix_rf = pd.DataFrame(confusion_matrix(dt.predict(X_test), y_test),
            columns = ['Predicted:Malicious', 'Predicted:Benign'],
            index = ['Actual:Malicious', 'Actual:Benign'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(dt.predict(X_test), y_test,
                            target_names =["0","1"]))

print('\n DECISION TREE CONFUSION MATRIX')
#plt.figure(figsize= (6,4))
sns.heatmap(confmatrix_rf, annot = True,fmt='d',cmap="YlGnBu")


# In[84]:


#Decision Tree 5-Fold Cross Validation 


# In[85]:


X = df[feature_col]
y = df['label'].values


# In[86]:


y_pred = cross_val_predict(dt, X, y, cv=5)
print('Cross-validated predictions: ', y_pred)

y_scores = cross_val_score(dt, X, y, cv=5, scoring='accuracy')
print('Cross-validated accuracy scores: ', y_scores)
print('Mean cross-validated accuracy scores: ', y_scores.mean())

y_scores_auc = cross_val_score(dt, X, y, cv=5, scoring='roc_auc')
print('Cross-validated auc scores: ', y_scores_auc)
print('Mean cross-validated auc scores: ', y_scores_auc.mean())


# # Identifying the Most Important Features for the Best Decision Tree Model

# In[87]:


df1.columns


# In[88]:


df2 = df11


# In[89]:


del df2['label']


# In[90]:


df2.columns


# In[91]:


importance = lab_clf5.feature_importances_
fn = df2.columns
fs=list(df2.columns)
fs2=pd.DataFrame(fs)
print("Feature importances:")
for i,v in enumerate(importance):
    print('Feature: %s, Score: %.5f' % (fn[i],v))


# In[92]:


#fs2


# In[93]:


fs3 = pd.DataFrame({'feature', 'score'})
fs3 = pd.DataFrame({'feature':fs,
                    'score': importance})


# In[94]:


fs3.sort_values(by=['score'],ascending=False).iloc[0:5]


# # Random Forest Classification

# In[95]:


df_rf= pd.read_csv('combined_dataset.csv')
df_rf1 = df_rf.drop('domain', axis=1)


# In[96]:


rfc = RandomForestClassifier(criterion = 'gini')
rfc.fit(X_train,y_train)
predicted_rfc = rfc.predict(X_test)
X = df_rf1[feature_col]
y = df_rf1['label'].values
fpr,tpr,thresh = roc_curve(y_test, predicted_rfc)    
roc_auc  = accuracy_score(y_test, predicted_rfc) 
y_pred_prob = rfc.predict_proba(X_test)[:, 1]
print("AUC Score: ", metrics.roc_auc_score(y_test, y_pred_prob))
print('Training Accuracy :',rfc.score(X_train, y_train))
print('Testing Accuracy :',rfc.score(X_test, y_test))


# In[97]:


c_matrix = confusion_matrix(y_test, predicted_rfc)
TP = c_matrix[1, 1]
TN = c_matrix[0, 0]
FP = c_matrix[0, 1]
FN = c_matrix[1, 0]
print(confusion_matrix(y_test,predicted_rfc))
print("False Positive Rate: ", FP / float(FP + TN))


# In[98]:


# Plot ROC curve for Random Forest

confmatrix_rf = pd.DataFrame(confusion_matrix(rfc.predict(X_test), y_test),
            columns = ['Predicted:Malicious', 'Predicted:Benign'],
            index = ['Actual:Malicious', 'Actual:Benign'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(rfc.predict(X_test), y_test,
                            target_names =["0","1"]))

print('\n RANDOM FOREST CONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(confmatrix_rf, annot = True,fmt='d',cmap="YlGnBu")


# In[99]:


from sklearn.ensemble import RandomForestClassifier

rfc2 = RandomForestClassifier(criterion = 'entropy')
rfc2.fit(X_train,y_train)
predicted_rfc2 = rfc2.predict(X_test)

X = df_rf1[feature_col]
y = df_rf1['label'].values

print("Accuracy with RF classifier:",accuracy_score(y_test, predicted_rfc2)) 
fpr,tpr,thresh = roc_curve(y_test, predicted_rfc2)    
roc_auc  = accuracy_score(y_test, predicted_rfc2) 

y_pred_prob = rfc2.predict_proba(X_test)[:, 1]
print("AUC Score: ", metrics.roc_auc_score(y_test, y_pred_prob))

print('Training Accuracy :',rfc2.score(X_train, y_train))
print('Testing Accuracy :',rfc2.score(X_test, y_test))


# In[100]:


from sklearn.ensemble import RandomForestClassifier

rfc3 = RandomForestClassifier(criterion = 'gini',max_depth = 5)
rfc3.fit(X_train,y_train)
predicted_rfc3 = rfc3.predict(X_test)

X = df_rf1[feature_col]
y = df_rf1['label'].values

print("Accuracy with RF classifier:",accuracy_score(y_test, predicted_rfc3)) 
fpr,tpr,thresh = roc_curve(y_test, predicted_rfc3)    
roc_auc  = accuracy_score(y_test, predicted_rfc3) 

y_pred_prob = rfc3.predict_proba(X_test)[:, 1]
print("AUC Score: ", metrics.roc_auc_score(y_test, y_pred_prob))

print('Training Accuracy :',rfc3.score(X_train, y_train))
print('Testing Accuracy :',rfc3.score(X_test, y_test))


# In[101]:


from sklearn.ensemble import RandomForestClassifier

rfc4 = RandomForestClassifier(criterion = 'entropy',max_depth = 12)
rfc4.fit(X_train,y_train)
predicted_rfc4 = rfc4.predict(X_test)

X = df_rf1[feature_col]
y = df_rf1['label'].values

print("Accuracy with RF classifier:",accuracy_score(y_test, predicted_rfc4)) 
fpr,tpr,thresh = roc_curve(y_test, predicted_rfc4)    
roc_auc  = accuracy_score(y_test, predicted_rfc4) 

y_pred_prob = rfc4.predict_proba(X_test)[:, 1]
print("AUC Score: ", metrics.roc_auc_score(y_test, y_pred_prob))

print('Training Accuracy :',rfc4.score(X_train, y_train))
print('Testing Accuracy :',rfc4.score(X_test, y_test))


# In[102]:


rfc = RandomForestClassifier()
param_grid = { 
    'n_estimators': [50, 150],
    'max_features': ['auto', 'sqrt', 'log2'],
    'criterion' :['gini', 'entropy']    
}
classifier_rf = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
classifier_rf.fit(X_train, y_train)
predicted_rf_classifier = classifier_rf.predict(X_test)
print("Accuracy: %.3f" % metrics.accuracy_score(y_test, predicted_rf_classifier))

classifier.best_params_


# In[103]:


y_pred = cross_val_predict(rfc, X, y, cv=5)
print('Cross-validated predictions: ', y_pred)

y_scores = cross_val_score(rfc, X, y, cv=5, scoring='accuracy')
print('Cross-validated accuracy scores: ', y_scores)
print('Mean cross-validated accuracy scores: ', y_scores.mean())

y_scores_auc = cross_val_score(rfc, X, y, cv=5, scoring='roc_auc')
print('Cross-validated auc scores: ', y_scores_auc)
print('Mean cross-validated auc scores: ', y_scores_auc.mean())


# # Logistic Regression

# In[104]:


df_lr= pd.read_csv('combined_dataset.csv')
df_lr1 = df_lr.drop('domain', axis=1)


# In[105]:


#logreg = LogisticRegression()
logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train,y_train)
predicted_lr = logreg.predict(X_test)

X = df_lr1[feature_col]
y = df_lr1['label'].values

print("Accuracy with Log Reg:" ,accuracy_score(y_test, predicted_lr ))
#print ("Conf matrix Log Reg:\n",confusion_matrix(y_test,predicted_lr ))
fpr,tpr,thresh = roc_curve(y_test,predicted_lr)
roc_auc = accuracy_score(y_test,predicted_lr )

print('Logreg Training Accuracy :',logreg.score(X_train, y_train))
print('Logreg Testing Accuracy :',logreg.score(X_test, y_test))

predicted_lr_prob = logreg.predict_proba(X_test)[:, 1]


print('Logistic Regression AUC: ', metrics.roc_auc_score(y_test, logreg.predict_proba(X_test)[:,1]))


# In[106]:


c_matrix = confusion_matrix(y_test, predicted_lr )
TP = c_matrix[1, 1]
TN = c_matrix[0, 0]
FP = c_matrix[0, 1]
FN = c_matrix[1, 0]
print(confusion_matrix(y_test,predicted_lr))
print("False Positive Rate: ", FP / float(FP + TN))


# In[107]:


predicted_lr  = logreg.predict(X_test)
print("Logistic regression accuracy on test data= %.3f" % (accuracy_score(predicted_lr , y_test)))
predicted_lr_train = logreg.predict(X_train)
print("Logistic regression accuracy on train data= %.3f" % (accuracy_score(predicted_lr_train, y_train)))

confmatrix_lr = pd.DataFrame(confusion_matrix(logreg.predict(X_test), y_test),
            columns = ['Predicted:Malicious', 'Predicted:Benign'],
            index = ['Actual:Malicious', 'Actual:Benign'])

print('\nCLASSIFICATION REPORT\n')
print(classification_report(logreg.predict(X_test), y_test,target_names =["0","1"]))

print('\n LOG REG CONFUSION MATRIX')
#plt.figure(figsize= (6,4))
sns.heatmap(confmatrix_lr, annot = True,fmt='d',cmap="YlGnBu")


# In[108]:



y_pred = cross_val_predict(logreg, X, y, cv=5)
print('Cross-validated predictions: ', y_pred)
y_scores = cross_val_score(logreg, X, y, cv=5, scoring='accuracy')
print('Cross-validated accuracy scores: ', y_scores)
print('Mean cross-validated accuracy scores: ', y_scores.mean())
y_scores_auc = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')
print('Cross-validated auc scores: ', y_scores_auc)
print('Mean cross-validated auc scores: ', y_scores_auc.mean())


# In[109]:


#Logreg #2

logreg2 = LogisticRegression(solver='lbfgs')
logreg2.fit(X_train,y_train)
predicted_lr2 = logreg2.predict(X_test)

X = df_lr1[feature_col]
y = df_lr1['label'].values

print("Accuracy with Log Reg:" ,accuracy_score(y_test, predicted_lr2))
#print ("Conf matrix Log Reg:\n",confusion_matrix(y_test,predicted_lr ))
fpr,tpr,thresh = roc_curve(y_test,predicted_lr2)
roc_auc = accuracy_score(y_test,predicted_lr2)

print('Logreg Training Accuracy :',logreg2.score(X_train, y_train))
print('Logreg Testing Accuracy :',logreg2.score(X_test, y_test))

predicted_lr2_prob = logreg2.predict_proba(X_test)[:, 1]


print('Logistic Regression AUC: ', metrics.roc_auc_score(y_test, logreg2.predict_proba(X_test)[:,1]))


# In[110]:


c_matrix = confusion_matrix(y_test, predicted_lr2)
TP = c_matrix[1, 1]
TN = c_matrix[0, 0]
FP = c_matrix[0, 1]
FN = c_matrix[1, 0]
print(confusion_matrix(y_test,predicted_lr2))
print("False Positive Rate: ", FP / float(FP + TN))


# In[111]:


# logreg #3
logreg3 = LogisticRegression(solver='newton-cg')
logreg3.fit(X_train,y_train)
predicted_lr3 = logreg3.predict(X_test)
X = df_lr1[feature_col]
y = df_lr1['label'].values
fpr,tpr,thresh = roc_curve(y_test,predicted_lr3)
roc_auc = accuracy_score(y_test,predicted_lr3)
print('Logreg Training Accuracy :',logreg3.score(X_train, y_train))
print('Logreg Testing Accuracy :',logreg3.score(X_test, y_test))
print('Logistic Regression AUC: ', metrics.roc_auc_score(y_test, 
        logreg3.predict_proba(X_test)[:,1]))


# In[112]:


c_matrix = confusion_matrix(y_test, predicted_lr3)
TP = c_matrix[1, 1]
TN = c_matrix[0, 0]
FP = c_matrix[0, 1]
FN = c_matrix[1, 0]
print(confusion_matrix(y_test,predicted_lr3))
print("False Positive Rate: ", FP / float(FP + TN))


# In[113]:


# logreg #4 

logreg4 = LogisticRegression(solver='sag')
logreg4.fit(X_train,y_train)
predicted_lr4 = logreg4.predict(X_test)

X = df_lr1[feature_col]
y = df_lr1['label'].values

fpr,tpr,thresh = roc_curve(y_test,predicted_lr4)
roc_auc = accuracy_score(y_test,predicted_lr4)

print('Logreg Training Accuracy :',logreg4.score(X_train, y_train))
print('Logreg Testing Accuracy :',logreg4.score(X_test, y_test))

predicted_lr4_prob = logreg4.predict_proba(X_test)[:, 1]

print(classification_report(logreg4.predict(X_test), y_test,target_names =["0","1"]))
print('Logistic Regression AUC: ', metrics.roc_auc_score(y_test, logreg4.predict_proba(X_test)[:,1]))


# In[114]:


# logreg #5

logreg5 = LogisticRegression(solver='saga')
logreg5.fit(X_train,y_train)
predicted_lr5 = logreg5.predict(X_test)

X = df_lr1[feature_col]
y = df_lr1['label'].values

fpr,tpr,thresh = roc_curve(y_test,predicted_lr5)
roc_auc = accuracy_score(y_test,predicted_lr5)

print('Logreg Training Accuracy :',logreg5.score(X_train, y_train))
print('Logreg Testing Accuracy :',logreg5.score(X_test, y_test))

predicted_lr5_prob = logreg5.predict_proba(X_test)[:, 1]

print(classification_report(logreg5.predict(X_test), y_test,target_names =["0","1"]))
print('Logistic Regression AUC: ', metrics.roc_auc_score(y_test, logreg5.predict_proba(X_test)[:,1]))


# In[115]:


c_matrix = confusion_matrix(y_test, predicted_lr5)
TP = c_matrix[1, 1]
TN = c_matrix[0, 0]
FP = c_matrix[0, 1]
FN = c_matrix[1, 0]
print(confusion_matrix(y_test,predicted_lr5))
print("False Positive Rate: ", FP / float(FP + TN))


# In[116]:


#LR model Grisdearch
logisticRegr = LogisticRegression()

#Fit
logisticRegr.fit(X_train, y_train)
predictions = logisticRegr.predict(X_test)




from sklearn.model_selection import GridSearchCV

#Instantiate
clf = LogisticRegression()

#Grid
parameter_grid = {'C': [0.01, 0.1, 1, 2, 10, 100], 'penalty': ['l1', 'l2']}

#Gridsearch
gridsearch = GridSearchCV(clf, parameter_grid)
gridsearch.fit(X_train, y_train);

#Get best hyperparameters
gridsearch.best_params_


# In[117]:


print('Logreg Training Accuracy :',logisticRegr.score(X_train, y_train))
print('Logreg Testing Accuracy :',logisticRegr.score(X_test, y_test))

#predicted_lr2_prob = predictions.predict_proba(X_test)[:, 1]


print('Logistic Regression AUC: ', metrics.roc_auc_score(y_test, logisticRegr.predict_proba(X_test)[:,1]))


# In[ ]:





# In[ ]:





# # SVM 

# In[118]:


df_svm = pd.read_csv('combined_dataset.csv')
df_svm1 = df_svm.drop('domain', axis=1)


X = df_svm1[feature_col]
y = df_svm1['label'].values

from sklearn import preprocessing
X = preprocessing.scale(X) 


# In[119]:


# Fit the SVM model #1
from sklearn.svm import SVC
svm_clf = SVC(kernel="linear", C=float("inf"), max_iter=4000)
svm_clf.fit(X_train, y_train);
y_predict_svm_train = svm_clf.predict(X_train)
print("SVM accuracy on train data = %.3f" % (accuracy_score(y_predict_svm_train, y_train)))
y_predict_svm_test = svm_clf.predict(X_test)
print("SVM accuracy on test data = %.3f" % (accuracy_score(y_predict_svm_test, y_test)))


# In[120]:


y_pred = cross_val_predict(svm_clf, X, y, cv=5)
print('Cross-validated predictions: ', y_pred)

y_scores = cross_val_score(svm_clf, X, y, cv=5, scoring='accuracy')
print('Cross-validated accuracy scores: ', y_scores)
print('Mean cross-validated accuracy scores: ', y_scores.mean())

y_scores_auc = cross_val_score(svm_clf, X, y, cv=5, scoring='roc_auc')
print('Cross-validated auc scores: ', y_scores_auc)
print('Mean cross-validated auc scores: ', y_scores_auc.mean())


# In[121]:


confmatrix_svm = pd.DataFrame(confusion_matrix(svm_clf.predict(X_test), y_test),
            columns = ['Predicted:Malicious', 'Predicted:Benign'],
            index = ['Actual:Malicious', 'Actual:Benign'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(svm_clf.predict(X_test), y_test,
                            target_names =["0","1"]))

print('\n SVM CONFUSION MATRIX')
#plt.figure(figsize= (6,4))
sns.heatmap(confmatrix_svm , annot = True,fmt='d',cmap="YlGnBu")


# In[122]:


# SVM Model Variation 2
from sklearn.svm import SVC
svm_clf2 = SVC(kernel="poly", C=float("inf"), max_iter=4500)
svm_clf2.fit(X_train, y_train);
predicted_svm2= svm_clf2.predict(X_test)
X = df_svm1[feature_col]
y = df_svm1['label'].values
X = preprocessing.scale(X) 
from sklearn.metrics import accuracy_score
# Compare test set predictions with ground truth labels
y_predict_svm_train = svm_clf2.predict(X_train)
print("SVM accuracy on train data = %.3f" % (accuracy_score(y_predict_svm_train, y_train)))
y_predict_svm_test = svm_clf2.predict(X_test)
print("SVM accuracy on test data = %.3f" % (accuracy_score(y_predict_svm_test, y_test)))


# In[123]:


# SVM Model Variation 3
from sklearn.svm import SVC
svm_clf3 = SVC(kernel="sigmoid", C=float("inf"), max_iter=5500)
svm_clf3.fit(X_train, y_train);
predicted_svm3= svm_clf3.predict(X_test)
# Compare test set predictions with ground truth labels
y_predict_svm_train = svm_clf3.predict(X_train)
print("SVM accuracy on train data = %.3f" % (accuracy_score(y_predict_svm_train, y_train)))
y_predict_svm_test = svm_clf3.predict(X_test)
print("SVM accuracy on test data = %.3f" % (accuracy_score(y_predict_svm_test, y_test)))


# In[124]:


#SVM Model Variation 4
from sklearn.svm import SVC
svm_clf4 = SVC(kernel="sigmoid", C=float("7"), max_iter=6000)
svm_clf4.fit(X_train, y_train);
predicted_svm4 = svm_clf4.predict(X_test)
from sklearn.metrics import accuracy_score
# Compare test set predictions with ground truth labels
y_predict_svm_train = svm_clf4.predict(X_train)
print("SVM accuracy on train data = %.3f" % (accuracy_score(y_predict_svm_train, y_train)))
y_predict_svm_test = svm_clf4.predict(X_test)
print("SVM accuracy on test data = %.3f" % (accuracy_score(y_predict_svm_test, y_test)))


# In[125]:


#SVM Model Variation 5 with GridSearchCV 
from sklearn import svm, metrics 
classifier = svm.SVC(kernel = 'linear', max_iter=5000)
classifier.fit(X_train, y_train)
predicted_svm_gs = classifier.predict(X_test)

X = df_svm1[feature_col]
y = df_svm1['label'].values


fpr,tpr,thresh = roc_curve(y_test,predicted_svm_gs )
roc_auc = accuracy_score(y_test,predicted_svm_gs )

print("Accuracy: %.3f" % metrics.accuracy_score(y_test, predicted_svm_gs))
predicted_svm_train = classifier.predict(X_train)
print("SVM GridSearch accuracy on train data= %.3f" % (accuracy_score(predicted_svm_train, y_train)))
predicted_svm  = classifier.predict(X_test)
print("SVM GridSearch accuracy on test data= %.3f" % (accuracy_score(predicted_svm , y_test)))


# In[126]:



c_matrix = confusion_matrix(y_test, predicted_svm_gs)
TP = c_matrix[1, 1]
TN = c_matrix[0, 0]
FP = c_matrix[0, 1]
FN = c_matrix[1, 0]
print(confusion_matrix(y_test,predicted_svm_gs))

print(TP)
print(TN)
print(FP)
print(FN)

print("False Positive Rate: ", FP / float(FP + TN))










# In[127]:


confmatrix_svm = pd.DataFrame(confusion_matrix(classifier.predict(X_test), y_test),
            columns = ['Predicted:Malicious', 'Predicted:Benign'],
            index = ['Actual:Malicious', 'Actual:Benign'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(classifier.predict(X_test), y_test,
                            target_names =["0","1"]))

print('\n SVM CONFUSION MATRIX')
#plt.figure(figsize= (6,4))
sns.heatmap(confmatrix_svm , annot = True,fmt='d',cmap="YlGnBu")


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

svm_gs = SVC()


param_grid= {'kernel': ('linear', 'rbf'),'C': [1, 10, 100]}

classifier_rf = GridSearchCV(svm_gs, param_grid)
classifier_rf.fit(X_train, y_train)
predicted_rf_classifier = classifier_rf.predict(X_test)

print("Accuracy: %.3f" % metrics.accuracy_score(y_test, predicted_svm_gs))
print('Best Criterion: %s' % classifier.best_estimator_.criterion)
print('Best Max_Depth: %s' % classifier.best_estimator_.max_depth)
print('Best Min_Samples_Split: %s' % classifier.best_estimator_.min_samples_split)
print('Best Min_Samples_Leaf: %s' % classifier.best_estimator_.min_samples_leaf)


# In[ ]:


from sklearn.svm import SVC
svm_clf8 = SVC(kernel="rbf", C=float("7"), max_iter=4000)
svm_clf8.fit(X_train, y_train);
predicted_svm8 = svm_clf8.predict(X_test)
from sklearn.metrics import accuracy_score
# Compare test set predictions with ground truth labels
y_predict_svm_train = svm_clf8.predict(X_train)
print("SVM accuracy on train data = %.3f" % (accuracy_score(y_predict_svm_train, y_train)))
y_predict_svm_test = svm_clf8.predict(X_test)
print("SVM accuracy on test data = %.3f" % (accuracy_score(y_predict_svm_test, y_test)))


# In[ ]:





# In[51]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
#Define a dictionary containing all hyperparameter values to try:
#svc = svm.SVC()
svc = svm.SVC(probability=True)
hyperparam_grid = {
    'kernel': ('linear', 'rbf'),
    'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
    'C': [1, 3, 5, 7, 9],
    'max_iter': [10]
}


# In[133]:


'''from sklearn.model_selection import GridSearchCV
classifier = GridSearchCV(svc, hyperparam_grid)
classifier.fit(X_train, y_train)
predicted_svm = classifier.predict(X_test)
#print("Accuracy: %.3f" % metrics.accuracy_score)
print("Accuracy: %.3f" % metrics.accuracy_score(y_test, predicted_svm))'''





#commented out due to long compute time


# In[128]:


classifier.best_params_


# In[129]:


y_pred_prob = classifier.predict_proba(X_test)[:, 1]
print("AUC Score: ", metrics.roc_auc_score(y_test, y_pred_prob))


# In[130]:


print('SVM GridSearch Training Accuracy :',classifier.score(X_train, y_train))
print('SVM GridSearchTesting Accuracy :',classifier.score(X_test, y_test))
print('SVM GridSearchAUC: ', metrics.roc_auc_score(y_test, classifier.predict_proba(X_test)[:,1]))


# In[131]:


#chart to plot ROC curve for each model

#1 Classification using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc = rfc.fit(X_train,y_train)
predicted_rfc = rfc.predict(X_test)
print("Accuracy with RF classifier:",accuracy_score(y_test, predicted_rfc)) 
fpr,tpr,thresh = roc_curve(y_test,predicted_rfc)      
roc_auc = accuracy_score(y_test,predicted_rfc)         # Calculate ROC AUC

# Plot ROC curve for Random Forest
plt.plot(fpr,tpr,'green',label = 'Random Forest')
plt.legend("Random Forest", loc='lower right')
plt.legend(loc='lower right')
#print("Conf matrix RF classifier:\n",confusion_matrix(y_test,predicted_rfc))  #  Generate confusion matrix

#2 Classification using logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg = logreg.fit(X_train,y_train)
predicted_lr = logreg.predict(X_test)
print("Accuracy with Log Reg:", accuracy_score(y_test, predicted_lr ))
#print ("Conf matrix Log Reg:\n",confusion_matrix(y_test,predicted_lr ))
fpr,tpr,thresh = roc_curve(y_test,predicted_lr )
roc_auc = accuracy_score(y_test,predicted_lr)

# Plot ROC curve for Logistic Regression
plt.plot(fpr,tpr,'blue',label = 'Logistic Regression')
plt.legend("Logistic Regression", loc='lower right')
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend(loc='lower right')

#3 Classification using SVM

classifier = svm.SVC(kernel = 'linear', max_iter=4000)
classifier.fit(X_train, y_train)
predicted_svm_gs = classifier.predict(X_test)
print("Accuracy with SVM-Linear:",accuracy_score(y_test,predicted_svm_gs))
fpr,tpr,thresh = roc_curve(y_test,predicted_svm_gs)
roc_auc = accuracy_score(y_test,predicted_svm_gs)

# Plot ROC curve for SVM-linear
plt.plot(fpr,tpr,'purple',label = 'SVM')
plt.legend("SVM", loc ='lower right')
plt.legend(loc ='lower right')
#print("Conf matrix SVM-linear:\n",confusion_matrix(y_test,predicted_svm_gs))

#Define a dictionary containing all hyperparameter values to try:

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)


predicted_dt_classifier = dt.predict(X_test)
print("Accuracy with Decision Tree: %.3f" % metrics.accuracy_score(y_test, predicted_dt_classifier))
fpr,tpr,thresh = roc_curve(y_test, dt.predict_proba(X_test)[:,1])
roc_auc        = accuracy_score(y_test, predicted_dt_classifier)
# Plot ROC curve for GS DT
plt.plot(fpr,tpr,'orange',label = 'Decision Tree')
plt.legend("DT", loc ='lower right')
plt.legend(loc ='lower right')
#print("Conf matrix DT:\n",confusion_matrix(y_test,predicted_gs_classifier))






plt.show()


# In[72]:


# Results accuracy score


# In[78]:



print("Accuracy with Decision Tree:              %.3f"   % accuracy_score(y_test, predicted_dt))
print("Accuracy with Decision Tree GridSearch:   %.3f"   % accuracy_score(y_test, predicted_dt_classifier))
print("Accuracy with RF classifier:              %.3f"   % accuracy_score(y_test, predicted_rfc)) 
print("Accuracy with Log Reg:                    %.3f"   % accuracy_score(y_test, predicted_lr))
print("Accuracy with SVM1:                       %.3f"   % accuracy_score(y_test, predicted_svm))
print("Accuracy with SVM2:                       %.3f"   % accuracy_score(y_test, predicted_svm2))
print("Accuracy with SVM3:                       %.3f"   % accuracy_score(y_test, predicted_svm3))
print("Accuracy with SVM4:                       %.3f"   % accuracy_score(y_test, predicted_svm4))
print("Accuracy with SVM5 w/ GridSearch:         %.3f"   % accuracy_score(y_test, predicted_svm_gs))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





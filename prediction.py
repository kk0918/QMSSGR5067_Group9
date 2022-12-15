#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:10:46 2022

@author: yangtianrun
"""
##Let's have a try and cut the dataset into a quarter; otherwise it will be too large to run the following model: 
test_final = final_rt_df.iloc[:5000,:]
    
#define the outpath 
out_path = "./QMSS_testing/" 

#Tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer

cv = TfidfVectorizer()
tfidf = cv.fit_transform(test_final.cleaned_review)
# sum_tf = pd.DataFrame(tfidf.sum(axis=0)) #computer explodes here 
tfidf_columns = cv.get_feature_names()
tfidf.columns = tfidf_columns 
tfidf


#Random Forest Regressor: the predictors is the tfidf sparse matrix, while the outcome is the dosmestic gross
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    tfidf,test_final["domestic gross"], test_size=0.2, random_state=42) 
 
from sklearn.ensemble import RandomForestRegressor
RegModel = RandomForestRegressor(n_estimators=100,criterion='mse')

#Printing all the parameters of Random Forest
print(RegModel)

#Creating the model on Training Data
RF=RegModel.fit(X_train,y_train)
prediction=RF.predict(X_test)

#Measuring Goodness of fit in Training data
from sklearn import metrics
print('R2 Value:',metrics.r2_score(y_train, RF.predict(X_train)))
 
#Measuring accuracy on Testing Data
print('Accuracy',100- (np.mean(np.abs((y_test - prediction) / y_test)) * 100))
 
#Plotting the feature importance for Top 10 most important columns

importance = RF.feature_importances_ 
#Predictors = test_final["domestic gross"]
#%matplotlib inline
#feature_importances = pd.Series(RF.feature_importances_, index=Predictors )
#importance.nlargest(10).plot(kind='barh')
#TestingDataResults=pd.DataFrame(data=X_test, columns=Predictors)



#We can also build a classifier to rate the positive and negative of the word
def grid_fun(df_in, label_in, test_size_in, path_in, xform_in, grid_d, cv_in):
    #TRAIN AN ALGO USING my_vec
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_fscore_support
    import pandas as pd 
    from sklearn.model_selection import GridSearchCV
    my_model = RandomForestClassifier(random_state=123)
    my_grid_model = GridSearchCV(my_model, param_grid=grid_d, cv=cv_in)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df_in, label_in, test_size=test_size_in, random_state=42)  #split the dataset to training and testing
    
    my_grid_model.fit(X_train, y_train)
    
    print ("Best perf", my_grid_model.best_score_)
    print ("Best perf", my_grid_model.best_params_)
    
    my_model = RandomForestClassifier(
        **my_grid_model.best_params_, random_state=123)
    
    #lets see how balanced the data is
    agg_cnts = pd.DataFrame(y_train).groupby('label')['label'].count()
    print (agg_cnts)
    
    my_model.fit(X_train, y_train)
    write_pickle(my_model, path_in, "rf")
    
    y_pred = my_model.predict(X_test)
    y_pred_proba = pd.DataFrame(my_model.predict_proba(X_test))
    y_pred_proba.columns = my_model.classes_
    
    metrics = pd.DataFrame(precision_recall_fscore_support(
        y_test, y_pred, average='weighted'))
    metrics.index = ["precision", "recall", "fscore", "none"]
    print (metrics)
    
    the_feats = read_pickle(path_in, xform_in)
    try:
        #feature importance
        fi = pd.DataFrame(my_model.feature_importances_)
        fi["feat_imp"] = the_feats.get_feature_names()
        fi.columns = ["feat_imp", "feature"]
        perc_propensity = len(fi[fi.feat_imp > 0]) / len(fi)
        print ("percent features that have propensity:", perc_propensity)
    except:
        print ("can't get features")
        pass
    return fi


#have to apply the chi function first in order to apply it to random forest
def chi_fun(df_in, label_in, name_in, out_path_in, num_feat):
    #chi-square
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import SelectKBest
    import pandas as pd
    feat_sel = SelectKBest(score_func=chi2, k=num_feat)
    dim_data = pd.DataFrame(feat_sel.fit_transform(df_in, label_in).toarray())
    feat_index = feat_sel.get_support(indices=True)
    feature_names = [df_in.columns[i] for i in feat_index]
    dim_data.columns = feature_names
    write_pickle(feat_sel, out_path_in, name_in)
    return dim_data


chi_data = chi_fun(tfidf,test_final["domestic gross"],"rf", out_path, 1000)


#the cross validation used here is 5
the_grid = {"n_estimators": [10, 100], "max_depth": [None, 10]}

#call the classifier function here
#rf_prediction = grid_fun(
  # chi_data, label_in, 0.2, out_path, "chi", the_grid, 5)
"""Module contains function used to make prediction"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def feature_importance(X, input_model, name): 
    """
    Function to create a graph of the most impactful features of a model using sklearns feature_imporances
    Impactfull features meaning the variable that most affects the models performance. 
    Only works on tree based models. i.e. Randomforest, Gradient boosting classifier,
    
    Arguments:
    X:                  The names of the features, only used to give the graph column names.
    input_model:        The fitted model for which the features are calculated.  
    name:               The title given to the plot
    """
        



    feat_importances = pd.Series(input_model.feature_importances_, index=X)

    fig = plt.figure(figsize=(15,5))
    ax = plt.gca()
    feat_importances.nlargest(20).plot(kind='bar', color=['blue','black'], edgecolor='black', linewidth= 3, ax=ax)
    plt.grid(visible=None, which='major', axis='y')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(name) 
    plt.close()
    
    return fig

def metric(X_train, X_test, y_train, y_test, real_synth, columns, plot=False):
    """
    Function trains and predicts either a classification or regression model and returns various evaluation metrics.
    Function also returns a feature importance plots showing the features that contributed the most to the prediction.
    
    Input:
    X_train: The features of the dataset used for the model training
    X_test: The features of the dataset used to make the prediction
    y_train: The target variable used for training
    y_test: the target variable used for evaluating
    real_synth: Either Real or synthetic to give the column the right header name
    columns: the columns of the features to plot
    """
    
    if y_train.count() < 20:
        return pd.DataFrame(data={'accuracy':np.nan, 'recall':np.nan, 'precision':np.nan, 'f1':np.nan}, index=real_synth).T, 0
    
    elif y_train.nunique() > 10:
        # Fitting a default regressor model
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=8)
    else:
        # Fitting a default classification model
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=8)

    model.fit(X_train,y_train)
    y_pred = [i for i in model.predict(X_test)]
      
    
    if plot == True:
        feat_imp_plot = feature_importance(input_model=model, X=columns, name=real_synth)
    else:
        feat_imp_plot = 0
       
    # Regression metrics
    if y_train.nunique() > 5:
        max_errorss = round(max_error(y_test, y_pred), 3)
        evs = round(explained_variance_score(y_test, y_pred), 3)
        r2 = round(r2_score(y_test, y_pred), 3)
        mse = round(mean_squared_error(y_test, y_pred), 3)
        learningcurves = pd.DataFrame(data={'max error':max_errorss, 
                                            'explained variance score':evs,
                                            'r2':r2, 'mean squared error':mse}, 
                                      index=real_synth).T
        
    # Classification metrics  
    else:
        accuracy = round(accuracy_score(y_test, y_pred), 3)
        try:
            recall = round(recall_score(y_test, y_pred), 3)
            precision = round(precision_score(y_test, y_pred), 3)
            f1 = round(f1_score(y_test, y_pred), 3)
        except:
            recall = round(recall_score(y_test, y_pred, average='macro', zero_division=0), 3)
            precision = round(precision_score(y_test, y_pred, average='macro', zero_division=0), 3)
            f1 = round(f1_score(y_test, y_pred, average='macro', zero_division=0), 3)


        learningcurves = pd.DataFrame(data={'accuracy':accuracy, 'recall':recall, 'precision':precision, 'f1':f1}, index=real_synth).T

    return learningcurves, feat_imp_plot


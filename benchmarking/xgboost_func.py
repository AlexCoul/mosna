from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def train_XGBoost(X, y):
    """
    Train an XGBoost model.

    Parameters
    ----------
    y: 
        np_array containing predictive variables.
    y : np_array, 
        Response / target variable.

    Returns
    -------
    model : xgboost
        Trained XGBoost model
    parameters : dict
        Best parameters, determined with grid search
    feature_importance: np_array of gain feature importance
    perfomance: metric of model perfomance on the test dataset
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)   
    parameters = {
        'min_child_weight': [x for x in range(1, 10, 2)],
        'gamma': [x for x in np.arange (0.5, 5.0, 0.5)],
        'subsample': [x for x in np.arange (0.5, 1.0, 0.1)],
        'colsample_bytree': [x for x in np.arange (0.5, 1.0, 0.2)],
        'max_depth': [x for x in range(2, 7, 2)]
        }
    grid = GridSearchCV(XGBClassifier(), parameters, cv=5, scoring = 'roc_auc')
    grid.fit(X_train, y_train)
    parameters = grid.best_params_
    
    model = XGBClassifier(**parameters)
    model = model.fit(X_train, y_train)
    feature_importance = model.feature_importances_
    
    pred_y = model.predict(X_test)
    perfomance = {}
    perfomance['f1_score'] =  f1_score(y_test, pred_y, average='macro')
    perfomance['accuracy'] =  accuracy_score(y_test, pred_y) 
    
    return model, parameters, feature_importance, perfomance


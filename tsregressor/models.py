import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class TSRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, pipeline=None,exog=None,lags=14,stationarity_lag=7,target=None):
        self.pipeline = pipeline
        self.exog = exog
        self.lags = lags
        self.stationarity_lag = stationarity_lag
        self.target = target
        
    def get_params(self, deep=True):
        # suppose this estimator has parameter "c"
        return {"pipeline":self.pipeline,"exog":self.exog,"lags":self.lags}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        
        # set params and train model
        
        
        
        
        if self.pipeline is None:
            # set up pipeline
            numeric_features = ['sepal length (cm)','sepal width (cm)']
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())])

            categorical_features = ['category']
            categorical_transformer = OneHotEncoder(handle_unknown='ignore')

            preprocessor = ColumnTransformer(
                transformers=[
                    ('target', target_transformer, self.target),
                    ('cat', categorical_transformer, categorical_features)])
            
            
            
            # final pipeline
            self.pipeline = Pipeline(steps =
                [('root',root_function),
                 ('stationarize',stationarize),
                 ('date_feats',date_feats),
                 ('lag_feats',lag_feats),
                 ('ensemble',ensemble),
                 ('unstationarize',unstationarize),
                 ('unroot',unroot_function)
                ])
        self.pipeline.fit(X, y)
        # trained model, ready for predicting
        return self
    
    def transform(self, X):
        check_is_fitted(self)
        X_new = self.pipeline.transform(X.copy())
        return X_new
    
    def predict(self, X):
        check_is_fitted(self)
        pred = self.pipeline.predict(X)
        return pred
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        X_new = self.transform(X)
        return X_new
    
    def fit_predict(self, X, y):
        self.fit(X, y)
        X_new = self.transform(X)
        return self.predict(X_new)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.ensemble import StackingRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class TSRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, horizon=None,exog=None,target=None,date=None):
        self.horizon = horizon
        self.exog = exog
        self.target = target
        self.date = date
        self.freq = None
        self.target_shift = None
        self.target_trans = pd.Series()
        self.clf = None
        self.feat_cols = None
        self.lmbda = None

    def get_params(self, deep=True):
        # suppose this estimator has parameter "c"
        return {"horizon":self.horizon,
                "exog":self.exog,
                "target":self.target,
                "date":self.date,
                'target_shift':self.target_shift,
                'target_trans':self.target_trans,
                'clf':self.clf,
                'lmbda':self.lmbda}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        #X, y = check_X_y(X, y)
        if not self.freq:
            self.freq = self.date.dt.freq

        # transform target
        y += 1
        temp,lmbda = boxcox(y)
        self.lmbda = lmbda
        target_trans = pd.Series(temp,index=self.date)
        target_trans = target_trans.reindex(pd.date_range(start=self.date.min(),periods=len(self.date.index)+self.horizon,freq=self.freq))
        self.target_shift = target_trans.shift(self.horizon)
        self.target_trans = (target_trans - self.target_shift).fillna(0)

        # transform features
        X_new = self.transform(X)
        self.feat_cols = X_new.columns

        # model
        self.clf = XGBRegressor()
        self.clf.fit(X_new.head(len(X.index)),self.target_trans.head(len(X.index)))

        return self

    def transform(self, X):
        #check_is_fitted(self)
        X_new = pd.DataFrame(index=self.date)
        X_new = X_new.reindex(pd.date_range(start=X_new.index.min(),periods=len(X_new.index)+self.horizon,freq=self.freq))
        X.index = self.date
        X = X.reindex(pd.date_range(start=self.date.min(),periods=len(self.date)+self.horizon,freq=self.freq))

        # Date Feats
        X_new['day_of_week'] = X_new.index.day_of_week
        X_new['day_of_month'] = X_new.index.day
        X_new['day_of_year'] = X_new.index.day_of_year
        X_new['week_of_year'] = X_new.index.weekofyear
        X_new['month_of_year'] = X_new.index.month

        # Lag Feats
        lag_cols = []
        if self.exog:
            for exog in self.exog:
                lag_cols.append(exog)
        for col in lag_cols:
            for x in range(self.horizon,self.horizon+self.horizon):
                X_new[col+'_Lag'+str(x)] = X[col].shift(x).fillna(0)
        if not self.target_trans.empty:
            if len(self.target_trans.index) < len(X_new.index):
                self.target_trans = self.target_trans.reindex(pd.date_range(start=X_new.index.min(),periods=len(X_new.index)+self.horizon,freq=self.freq))
            for x in range(self.horizon,self.horizon+self.horizon):
                X_new['target_trans_Lag'+str(x)] = self.target_trans.shift(x).fillna(0)

        return X_new

    def predict(self, X_trans):
        #check_is_fitted(self)

        # predict on transformed target
        pred_trans = self.clf.predict(X_trans[self.feat_cols])

        # untransform
        if len(X_trans.index) > len(self.target_shift.index):
            self.target_shift = self.target_shift.reindex(pd.date_range(start=self.target_shift.index.min(),periods=len(X_trans.index),freq=self.freq))
        temp = pred_trans + self.target_shift
        y_pred = inv_boxcox(temp,self.lmbda)
        y_pred -= 1

        return y_pred

    def fit_transform(self, X, y):
        self.fit(X, y)
        X_new = self.transform(X)
        return X_new

    def fit_predict(self, X, y):
        self.fit(X, y)
        X_new = self.transform(X)
        return self.predict(X_new)

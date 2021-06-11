import pandas as pd
import numpy as np
from xgboost import XGBRegressor

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class TSRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, horizon=None,pipeline=None,exog=None,stationarity_lag=7,target=None,date=None,lags=None):
        self.horizon = horizon
        self.pipeline = pipeline
        self.exog = exog
        self.stationarity_lag = stationarity_lag
        self.target = target
        self.date = date
        self.lags = lags

    def get_params(self, deep=True):
        # suppose this estimator has parameter "c"
        return {"horizon":self.horizon,
                "pipeline":self.pipeline,
                "exog":self.exog,
                "stationarity_lag":self.stationarity_lag,
                "date":self.date,
                'lags':lags}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        # set params and train model

        # build pipeline
        if self.pipeline is None:
            root_function_trans = ColumnTransformer(transformers=[('root_function',RootFunction, self.target)])
            root_function = Pipeline(steps=[('root_function',root_function_trans)])

            stationarize_trans = ColumnTransformer(transformers=[('stationarize',Stationarize(stationarity_lag=self.stationarity_lag,horizon=self.horizon), self.target)])
            stationarize = Pipeline(steps=[('stationarize',stationarize_trans)])

            date_feats_trans = ColumnTransformer(transformers=[('date_feats',DateFeats(date=self.date), self.date)])
            date_feats = Pipeline(steps=[('date_feats',date_feats_trans)])

            lag_cols = [self.target]
            if self.exog:
                lag_cols.append(self.exog)
            lag_feats_trans = ColumnTransformer(transformers=[('lag_feats',LagFeats(exog=lag_cols,lags=self.lags,horizon=self.horizon),lag_cols)])
            lag_feats = Pipeline(steps=[('lag_feats',lag_feats_trans)])

            ensemble_trans = StackingRegressor(estimators=[('xgboost1',XGBRegressor),('xgboost2',XGBRegressor),('xgboost3',XGBRegressor)])
            ensemble = Pipeline(steps=[('ensemble',ensemble_trans)])

            unstationarize_trans = ColumnTransformer(transformers=[('unstationarize',Unstationarize(stationarity_lag=self.stationarity_lag,horizon=self.horizon),self.target)])
            unstationarize = Pipeline(steps=[('unstationarize',unstationarize_trans)])

            unroot_function_trans = ColumnTransformer(transformers=[('unroot_function',UnrootFunction, self.target)])
            unroot_function = Pipeline(steps=[('unroot_function',unroot_function_trans)])
            # final pipeline
            self.pipeline = Pipeline(steps =
                [('root_function',root_function),
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

class RootFunction(BaseEstimator, TransformerMixin):
    def __init__(self,target=None):
        self.target = target

    def fit(self, X, y):
        # find params
        return self

    def transform(self, X, y=None):
        # apply function
        root_function = lambda x: np.sign(x) * np.power(abs(x), .5)
        X_new = X.apply(root_function)
        return X_new

class UnrootFunction(BaseEstimator, TransformerMixin):
    def __init__(self,target=None):
        self.target = target

    def fit(self, X, y):
        # find params
        return self

    def transform(self, X, y=None):
        # apply function
        unroot_function = lambda x: np.sign(x) * np.power(abs(x), 2)
        X_new = X.apply(unroot_function)
        return X_new

class Stationarize(BaseEstimator, TransformerMixin):
    def __init__(self,stationarity_lag=None,horizon=None):
        self.stationarity_lag = stationarity_lag
        self.horizon = horizon

    def fit(self, X, y):
        # find params
        if not self.stationarity_lag:
            self.stationarity_lag = self.horizon
        return self

    def transform(self, X, y=None):
        # apply function
        if not self.stationarity_lag:
            X_new = X - X.shift(self.stationarity_lag)
        else:
            X_new = None
        return X_new

class Unstationarize(BaseEstimator, TransformerMixin):
    def __init__(self,stationarity_lag=None,horizon=None):
        self.stationarity_lag = stationarity_lag
        self.horizon = horizon

    def fit(self, X, y):
        # find params
        return self

    def transform(self, X, y=None):
        # apply function
        if (not self.stationarity_lag) & (not self.horizon):
            X_new = X + X.shift(self.stationarity_lag - self.horizon)
        else:
            X_new = None
        return X_new

class DateFeats(BaseEstimator, TransformerMixin):
    def __init__(self,date=None):
        self.date = date

    def fit(self, X, y):
        # find params
        return self

    def transform(self, X, y=None):
        # apply function
        X_new = pd.DataFrame()
        X_new['day_of_week'] = X[self.date].dt.day_of_week
        X_new['day_of_month'] = X[self.date].dt.day
        X_new['day_of_year'] = X[self.date].dt.day_of_year
        X_new['weekofyear'] = X[self.date].dt.weekofyear
        X_new['month_of_year'] = X[self.date].dt.month
        return X_new


class LagFeats(BaseEstimator, TransformerMixin):
    def __init__(self,exog=None,lags=None,horizon=None):
        self.lags = lags
        self.horizon = horizon
        self.exog = exog

    def fit(self, X, y):
        # find params
        if not self.lags:
            if self.horizon:
                self.lags = self.horizon
        return self

    def transform(self, X, y=None):
        # apply function
        X_new = pd.DataFrame()
        if self.exog:
            for i, exog in enumerate(self.exog):
                for j, lag in enumerate(lags):
                    lag_col = exog + "_Lag" + str(lag)
                    X_new[lag_col] = X[exog].shift(lag)

        return X_new

import os
import joblib
import pandas as pd
import numpy as np

from src.pipeline import fetch_timeseries, prep_univariate, prep_prophet

from pydantic import BaseModel
from fbprophet import Prophet
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import ForecastingHorizon

UNIVARIATE_MODELS = ['auto_arima', 'prophet']
MULTIVARIATE_MODELS = []


class CovidInput(BaseModel):
    pred_start: str = '2020-11-01'
    type: str = 'auto_arima'
    horizon: int = 7


class CovidModel:

    def __init__(self, pred_start, type, horizon, code='DE'):
        # Build model name according to input parameters
        self.model_fname = f'model/{type}/{pred_start}_{horizon}/{type}.pkl'
        # Load model from directory if it has been trained already, otherwise train it
        try:
            self.model = joblib.load(self.model_fname)
        except Exception as _:
            print(f'Creating new {type} model with test period starting from {pred_start} '
                  f'and prediction horizon of {horizon} days')

            # Fetch timeseries from STATWORX API
            self.input = fetch_timeseries(code=code)

            if type in UNIVARIATE_MODELS:
                # Prepare univariate timeseries
                self.y_train, self.y_test = prep_univariate(self.input, pred_start, horizon)

                if type == 'prophet':
                    self.y_train, self.y_test = prep_prophet(self.y_train, self.y_test)

            elif self.model in MULTIVARIATE_MODELS:
                pass

            # Train and fit selected model
            self.forecaster = self.train(type)
            self.model = self.fit()

            # Create directories for saving the model
            os.makedirs(f'model/{type}/{pred_start}_{horizon}/')

            # Dump model pkl file
            joblib.dump(self.model, self.model_fname)

    def train(self, type):
        if type == 'auto_arima':
            forecaster = AutoARIMA(sp=7, suppress_warnings=True)
        elif type == 'prophet':
            forecaster = Prophet()
        else:
            raise NotImplementedError(f'Model {type} is currently not implemented')

        return forecaster

    def fit(self):
        model = self.forecaster
        model.fit(self.y_train)

        return model

    def predict(self, pred_dates, type):
        fh = np.arange(len(pred_dates)) + 1

        if type == 'prophet':
            future = self.model.make_future_dataframe(periods=len(pred_dates), include_history=False)
            y_pred = self.model.predict(future).yhat
        else:
            y_pred = self.model.predict(fh)

        return y_pred


if __name__ == '__main__':

    model = CovidModel(pred_start='2020-11-01',
                       type='prophet',
                       horizon=7)

    pred_dates = pd.date_range(start='2020-11-01', periods=7).to_period('D')
    forecast = model.predict(pred_dates, 'prophet').to_list()
    print(forecast)

# models
# AutoARIMA
# XGB
# RFRegression
# DeepAR
# TBATS


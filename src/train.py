import os
import joblib

from src.pipeline import fetch_timeseries, prep_timeseries

from pydantic import BaseModel
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import ForecastingHorizon


class CovidInput(BaseModel):
    pred_start: str = '2020-11-01'
    type: str = 'auto_arima'
    horizon: int = 7


class CovidModel:

    def __init__(self, pred_start, type, horizon, code='DE'):
        # Build model name according to input parameters
        self.model_fname = f'model/{type}/{pred_start}_{horizon}/auto_arima.pkl'
        # Load model from directory if it has been trained already, otherwise train it
        try:
            self.model = joblib.load(self.model_fname)
        except Exception as _:
            print(f'Creating new {type} model with test period starting from {pred_start} '
                  f'and prediction horizon of {horizon} days')

            # Create directories for saving the model
            os.makedirs(f'model/{type}/{pred_start}_{horizon}/')

            # Fetch timeseries from STATWORX API
            self.input = fetch_timeseries(code=code)

            # Prepare timeseries
            self.y_train, self.y_test = prep_timeseries(self.input, pred_start, horizon)

            # Train and fit selected model
            self.forecaster = self.train(type)
            self.model = self.fit()

            # Dump model pkl file
            joblib.dump(self.model, self.model_fname)

    def train(self, type):
        if type == 'auto_arima':
            forecaster = AutoARIMA(sp=7, suppress_warnings=True)
        else:
            raise NotImplementedError(f'Model {type} is currently not implemented')

        return forecaster

    def fit(self):
        print(self.y_train)
        model = self.forecaster
        model.fit(self.y_train)

        return model

    def predict(self, pred_dates):
        fh = ForecastingHorizon(pred_dates, is_relative=False)
        y_pred = self.model.predict(fh)

        return y_pred



# models
# AutoARIMA
# XGB
# RFRegression
# DeepAR
# Prophet
# TBATS


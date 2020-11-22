import os
import joblib
import pandas as pd
import numpy as np

from src.pipeline import fetch_timeseries, prep_univariate, prep_prophet

from pydantic import BaseModel
from fbprophet import Prophet
from sktime.forecasting.arima import AutoARIMA

from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions


UNIVARIATE_MODELS = ['auto_arima', 'prophet', 'deepar']
MULTIVARIATE_MODELS = []


class CovidInput(BaseModel):
    pred_start: str = '2020-11-01'
    type: str = 'auto_arima'
    horizon: int = 7


class CovidModel:

    def __init__(self, pred_start, type, horizon, code='DE', freq='1D'):
        self.horizon = horizon
        self.freq = freq
        self.type = type
        # Build model name according to input parameters
        self.model_fname = f'model/{type}/{pred_start}/{type}.pkl'

        if type not in UNIVARIATE_MODELS or MULTIVARIATE_MODELS:
            raise NotImplementedError(f'Model called {type} not implemented')

        # Fetch timeseries from STATWORX API
        try:
            self.input = fetch_timeseries(code=code)
        except ImportError:
            print(f'Not able to fetch COVID-19 data of country {code} from STATWORX API')

        # Load model from directory if it has been trained already, otherwise train it
        try:
            # Get test data for gluonts models
            if self.type == 'deepar':
                _, self.y_test = prep_univariate(self.input, pred_start, horizon, type, self.freq)
            self.model = joblib.load(self.model_fname)
        except Exception as _:
            print(f'Creating new {type} model with test period starting from {pred_start} '
                  f'and prediction horizon of {horizon} days')

            if type in UNIVARIATE_MODELS:
                # Prepare univariate timeseries
                self.y_train, self.y_test = prep_univariate(self.input, pred_start, horizon, type, self.freq)

                if type == 'prophet':
                    self.y_train, self.y_test = prep_prophet(self.y_train, self.y_test)

            elif self.model in MULTIVARIATE_MODELS:
                pass

            # Train and fit selected model
            self.forecaster = self.train()
            self.model = self.fit()

            # Create directories for saving the model
            os.makedirs(f'model/{type}/{pred_start}/')

            # Dump model pkl file
            joblib.dump(self.model, self.model_fname)

    def train(self):
        if self.type == 'auto_arima':
            forecaster = AutoARIMA(sp=7, suppress_warnings=True)
        elif self.type == 'prophet':
            forecaster = Prophet()
        elif self.type == 'deepar':
            forecaster = DeepAREstimator(
                prediction_length=self.horizon,
                freq=self.freq,
                trainer=Trainer(ctx="cpu",
                                epochs=15,
                                learning_rate=1e-3,
                                num_batches_per_epoch=100
                                )
            )
        else:
            raise NotImplementedError(f'Model {type} is currently not implemented')

        return forecaster

    def fit(self):
        model = self.forecaster

        if self.type == 'deepar':
            model = model.train(self.y_train)
        else:
            model.fit(self.y_train)

        return model

    def predict(self, pred_dates):
        fh = np.arange(len(pred_dates)) + 1

        if self.type == 'deepar':
            _, ts_it = make_evaluation_predictions(
                dataset=self.y_test,
                predictor=self.model,
                num_samples=len(pred_dates),
            )

            y_pred = list(ts_it)[0][0].to_numpy()[-self.horizon:].tolist()

        elif self.type == 'prophet':
            future = self.model.make_future_dataframe(periods=len(pred_dates), include_history=False)
            y_pred = self.model.predict(future).yhat.tolist()
        else:
            y_pred = self.model.predict(fh).tolist()

        return y_pred


if __name__ == '__main__':

    model = CovidModel(pred_start='2020-11-01',
                       type='prophet',
                       horizon=7)

    pred_dates = pd.date_range(start='2020-11-01', periods=7).to_period('D')
    forecast = model.predict(pred_dates)
    print(forecast)



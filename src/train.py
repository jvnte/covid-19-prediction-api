import os
import joblib
import pandas as pd
import numpy as np

from src.pipeline import fetch_timeseries, prep_univariate, prep_prophet

from pydantic import BaseModel
from fbprophet import Prophet
from sktime.forecasting.arima import AutoARIMA

from gluonts.dataset.util import to_pandas
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions


UNIVARIATE_MODELS = ['auto_arima', 'prophet', 'deepar']
MULTIVARIATE_MODELS = []


class CovidInput(BaseModel):
    country: str = 'Germany'
    pred_start: str = '2020-11-01'
    type: str = 'auto_arima'
    horizon: int = 7


class CovidModel:

    def __init__(self, pred_start, type, horizon, country, freq='1D'):
        self.horizon = horizon
        self.freq = freq
        self.type = type
        self.country = country
        # Build model name according to input parameters
        self.model_fname = f'model/{type}/{pred_start}/{country}/{type}.pkl'

        if type not in UNIVARIATE_MODELS or MULTIVARIATE_MODELS:
            raise NotImplementedError(f'Model called {type} not implemented')

        # Fetch timeseries from STATWORX API
        try:
            self.input = fetch_timeseries(country=self.country)
        except ImportError:
            print(f'Not able to fetch COVID-19 data of {self.country} from API')

        self.y_train, self.y_test = prep_univariate(self.input, pred_start, horizon, type, self.freq)

        # Load model from directory if it has been trained already, otherwise train it
        try:
            self.model = joblib.load(self.model_fname)
        except Exception as _:
            print(f'Creating new {type} model for {country} with test period starting from {pred_start} '
                  f'and prediction horizon of {horizon} days')

            # Prepare prophet data
            if type == 'prophet':
                self.y_train, self.y_test = prep_prophet(self.y_train, self.y_test)

            # Train and fit selected model
            self.forecaster = self.build_forecaster()
            self.model = self.train()

            # Create directories for saving the model
            os.makedirs(f'model/{type}/{pred_start}/{country}/')

            # Dump model pkl file
            joblib.dump(self.model, self.model_fname)

    def build_forecaster(self):
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

    def train(self):
        model = self.forecaster

        if self.type == 'deepar':
            model = model.train(self.y_train)
        else:
            model.fit(self.y_train)

        return model

    def predict(self, pred_dates):
        fh = np.arange(len(pred_dates)) + 1

        if self.type == 'deepar':
            forecast_it, _ = make_evaluation_predictions(
                dataset=self.y_test,
                predictor=self.model,
                num_samples=len(pred_dates),
            )

            y_pred = list(forecast_it)[0].mean.tolist()

        elif self.type == 'prophet':
            future = self.model.make_future_dataframe(periods=len(pred_dates), include_history=False)
            y_pred = self.model.predict(future).yhat.tolist()

        else:
            y_pred = self.model.predict(fh).tolist()

        return y_pred

    def prepare_output(self, forecasts, horizon):
        # Get forecasts depending on horizon
        forecasts = forecasts[-horizon:]

        if self.type == 'deepar':
            # From iterator to pandas
            train = to_pandas(next(iter(self.y_train)))
            test = to_pandas(next(iter(self.y_test)))

            date = test.index.astype(str).tolist()

            target = test.tolist()

            prediction = np.full(len(train.index), np.nan).tolist()
            prediction.extend(forecasts)

        else:
            if self.type == 'prophet':
                if not isinstance(self.y_train, pd.Series):
                    self.y_train = self.y_train.set_index('ds').iloc[:, 0]
                if not isinstance(self.y_test, pd.Series):
                    self.y_test = self.y_test.set_index('ds').iloc[:, 0]

            date = self.y_train.index.astype(str).tolist()
            date.extend(self.y_test.index.astype(str).tolist())

            target = self.y_train.tolist()
            target.extend(self.y_test.tolist())

            prediction = np.full(len(self.y_train.index), np.nan).tolist()
            prediction.extend(forecasts)

        return date, target, prediction


if __name__ == '__main__':

    pred_start = '2020-11-29'
    model = 'deepar'
    horizon = 14

    model = CovidModel(pred_start=pred_start,
                       type=model,
                       horizon=horizon)

    pred_dates = pd.date_range(start=pred_start, periods=horizon).to_period('D')
    forecast = model.predict(pred_dates)
    print(forecast)

    date, target, prediction = model.prepare_output(forecast, horizon)




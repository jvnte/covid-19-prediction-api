import requests
import json
import pandas as pd

from gluonts.dataset.common import ListDataset
from sktime.forecasting.model_selection import temporal_train_test_split
from datetime import datetime, timedelta


def fetch_timeseries(code: str = 'DE'):
    # POST to API
    payload = {'code': code}
    url = 'https://api.statworx.com/covid'
    response = requests.post(url=url, data=json.dumps(payload))

    # Convert to data frame
    df = pd.DataFrame.from_dict(json.loads(response.text))

    return df


def prep_univariate(df, pred_start, horizon, type, freq):
    if type == 'deepar':
        ts = df.cases.to_numpy()
        start = pd.Timestamp("31-12-2019", freq=freq)

        # train dataset: cut the last window of length "prediction_length", add "target" and "start" fields
        y_train = ListDataset([{'target': ts[:-horizon], 'start': start}],
                              freq=freq)
        # test dataset: use the whole dataset, add "target" and "start" fields
        y_test = ListDataset([{'target': ts, 'start': start}],
                             freq=freq)

    else:
        # Get covid cases
        df = df[['date', 'cases']]

        # Convert to Series object and set index
        df = df.set_index('date').iloc[:, 0]

        # Convert index to period index
        df.index = pd.to_datetime(df.index).to_period('D')

        # Slice timeseries
        cut_off = (datetime.strptime(pred_start, "%Y-%m-%d")
                   - timedelta(days=1)
                   + timedelta(days=horizon)).strftime('%Y-%m-%d')

        df = df.loc[:cut_off]

        # Make temporal split
        y_train, y_test = temporal_train_test_split(df, test_size=horizon)

    return y_train, y_test


def prep_prophet(y_train, y_test):
    # Transform Series to Dataframe and rename columns
    y_train = y_train.to_frame().reset_index().rename(columns={'date': 'ds', 'cases': 'y'})
    y_test = y_test.to_frame().reset_index().rename(columns={'date': 'ds', 'cases': 'y'})

    # Transform date column to datetime
    y_train['ds'] = pd.to_datetime(y_train['ds'].astype('str'))
    y_test['ds'] = pd.to_datetime(y_test['ds'].astype('str'))

    return y_train, y_test

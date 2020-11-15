import requests
import json
import pandas as pd

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


def prep_timeseries(df, pred_start, horizon):
    # Get covid cases
    df = df[['date', 'cases']]

    # Convert to Series object and set index
    df = df.set_index('date').iloc[:,0]

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

import requests
import json
import os
import base64

import streamlit as st
import pandas as pd
import plotly.express as px

from pathlib import Path
from datetime import datetime, timedelta


@st.cache
def fetch_from_api(pred_start, type, horizon, country):
    payload = {
        "country": country,
        "pred_start": pred_start,
        "type": type,
        "horizon": horizon
    }

    url = 'http://127.0.0.1:8000/predict'

    try:
        response = requests.post(url=url, data=json.dumps(payload))
    except RuntimeError:
        print('API not accessible')

    data = pd.DataFrame.from_dict(json.loads(response.text))

    return data


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """

    csv = df.dropna().to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download csv file</a>'


COUNTRIES = ['Germany', 'China', 'US']
MODELS = {'Auto ARIMA': 'auto_arima',
          'Prophet': 'prophet',
          'DeepAR': 'deepar'}

MIN_DATE = datetime.strptime('2020-06-01', '%Y-%m-%d')
TODAY = datetime.now()

if __name__ == '__main__':

    st.header('Welcome to COVID-19 AutoML Dashboard')

    country = st.selectbox('Select the country', list(COUNTRIES))
    model = st.selectbox('Select the model to be trained', list(MODELS.keys()))
    horizon = st.slider('Set prediction horizon in days', min_value=1, value=14)

    # Dynamically adjust maximum prediction date depending on chosen horizon
    pred_start = st.date_input('Select the prediction start date',
                               min_value=MIN_DATE,
                               max_value=TODAY - timedelta(days=horizon + 3),
                               value=TODAY - timedelta(days=horizon + 3))

    model_path = Path(f'model/{MODELS.get(model)}/{pred_start}/{country}/{MODELS.get(model)}.pkl')

    if model_path.is_file():
        st.info(f'There is a pretrained {model} model available with prediction start {pred_start} for {country}.')
    else:
        st.warning(f'There is no pretrained {model} model available with prediction start {pred_start} for {country}. '
                   f'Making an API call results in training a new {model} model.')

    if st.button('Get predictions'):
        # Fetch from API
        df = fetch_from_api(pred_start=pred_start.strftime('%Y-%m-%d'),
                            type=MODELS.get(model),
                            horizon=horizon,
                            country=country)

        # Display only last 60 days
        df = df.tail(60)

        # From wide to long
        df = pd.melt(df, id_vars=['date'], value_vars=['target', 'prediction'])

        # Set maximum y-axis value
        max_y = df['value'].max() + df['value'].max() * 0.1

        # Create lineplot
        fig = px.line(df, x="date", y="value", color='variable')
        fig.update_yaxes(range=[0, max_y])
        fig.add_shape(type='line', x0=pred_start, y0=0, x1=pred_start, y1=max_y)

        st.plotly_chart(fig)

        st.markdown(get_table_download_link(df), unsafe_allow_html=True)

import requests
import json
import datetime
import streamlit as st
import pandas as pd


@st.cache
def fetch_from_api(pred_start, type, horizon):
    payload = {
        "pred_start": pred_start,
        "type": type,
        "horizon": horizon
    }

    url = 'http://127.0.0.1:8000/predict'

    try:
        response = requests.post(url=url, data=json.dumps(payload))
    except RuntimeError:
        print('API not accessible')

    df = pd.DataFrame.from_dict(json.loads(response.text)).set_index('date')

    return df


MODELS = {'Auto ARIMA': 'auto_arima',
          'Prophet': 'prophet',
          'DeepAR': 'deepar'}

MIN_DATE = datetime.datetime.strptime('2020-06-01', '%Y-%m-%d')
MAX_DATE = datetime.datetime.now()

if __name__ == '__main__':

    st.header('Welcome to COVID-19 AutoML Dashboard')

    model = st.selectbox('Select the model to be trained', list(MODELS.keys()))
    pred_start = st.date_input('Select the prediction date', min_value=MIN_DATE, max_value=MAX_DATE)
    horizon = st.slider('Set prediction horizon in days', min_value=1)

    if st.button('Get predictions'):
        df = fetch_from_api(pred_start=pred_start.strftime('%Y-%m-%d'),
                            type=MODELS.get(model),
                            horizon=horizon)

        st.line_chart(df)
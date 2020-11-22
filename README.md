## Welcome to COVID-19 AutoML API	

A API that makes predictions for COVID-19 cases in Germany. The data is fetched using [STATWORX COVID-19 API](https://github.com/STATWORX/covid-19-api)
and [FastAPI](https://fastapi.tiangolo.com/) is used as web framework for building the API. There is also a small
[Streamlit](https://www.streamlit.io/) dashboard that allows you to easily interact with the API.

## Currently implemented models

The following models are currently implemented:

- [AutoARIMA](https://www.sktime.org/en/latest/modules/auto_generated/sktime.forecasting.arima.AutoARIMA.html#sktime.forecasting.arima.AutoARIMA)
- [Prophet](https://facebook.github.io/prophet/)
- [DeepAR](https://ts.gluon.ai/api/gluonts/gluonts.model.deepar.html)

## Run Dashboard and API locally

To host the dashboard and API locally run the following command within project root directory

```shell
make setup
make run
```

These commands install all the required packages and start the API and dashboard. The dashboard should start in a new 
browser tab at http://localhost:8501/, while the API is running at http://localhost:8000/. To interact with the API only 
check out the next section.

## Run API only

Within the project root directory run the API within you CLI as follows:

```shell
uvicorn api:api --reload
```

Access FastAPI UI by open your browser at http://127.0.0.1:8000/docs. Open the POST method tab and click on *Try it out*.
You can manipulate the JSON request body as you desire. Clicking on *Execute* does the following:
 
- Make an API call to [STATWORX COVID-19 API](https://github.com/STATWORX/covid-19-api) to fetch the latest data
- Check whether the desired model has already been trained at *pred_start* and trains it if is has not been trained
- Makes prediction in your desired prediction horizon

The default JSON request body:
```json
{
  "pred_start": "2020-11-01",
  "type": "auto_arima",
  "horizon": 7
}
```





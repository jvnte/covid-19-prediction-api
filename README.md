## Welcome to COVID-19 AutoML API	

An API that allows you to easily generate predictions for worldwide COVID-19 cases. The data is fetched using [pomber COVID-19 API](https://github.com/pomber/covid19)
and [FastAPI](https://fastapi.tiangolo.com/) is used as web framework for building the API. There is also a small
[Streamlit](https://www.streamlit.io/) dashboard that allows you to easily interact with the API.

## Currently implemented models

The following models are currently implemented for univariate forecasting:

- [AutoARIMA](https://www.sktime.org/en/latest/modules/auto_generated/sktime.forecasting.arima.AutoARIMA.html#sktime.forecasting.arima.AutoARIMA)
- [Prophet](https://facebook.github.io/prophet/)
- [DeepAR](https://ts.gluon.ai/api/gluonts/gluonts.model.deepar.html)

## Run API only

Within the project root directory run the API within you CLI as follows:

```shell
uvicorn api:api --reload
```

Access FastAPI UI by open your browser at http://127.0.0.1:8000/docs. Open the POST method tab and click on `Try it out`.
You can manipulate the JSON request body as you desire. Clicking on `Execute` does the following:

- Make an API call to [pomber COVID-19 API](https://github.com/pomber/covid19) to fetch the latest data
- Check whether the desired model has already been trained at *pred_start* and trains it if is has not been trained
- Makes prediction in your desired prediction horizon

The default JSON request body:
```json
{
  "country": "Germany",
  "pred_start": "2020-11-01",
  "type": "auto_arima",
  "horizon": 7
}
```

Check out all avaliable countries at https://pomber.github.io/covid19/timeseries.json

## Run Dashboard and API locally

To host the dashboard and API locally run the following command within project root directory

```shell
make setup
make run
```

These commands install all the required packages and start the API and dashboard. The dashboard should start in a new 
browser tab at http://localhost:8501/, while the API is running at http://localhost:8000/. To interact with the API only 
check out the previous section.

## License

The data is under [CSSEGISandData/COVID-19](https://github.com/CSSEGISandData/COVID-19/) terms of use.



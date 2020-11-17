## Welcome to COVID-19 AutoML API	

A API that makes predictions for COVID-19 cases in Germany. The data is fetched using [STATWORX COVID-19 API](https://github.com/STATWORX/covid-19-api)
and [FastAPI](https://fastapi.tiangolo.com/) is used as web framework for building the API.

## Run API locally

Within the project root directory run the API within you CLI as follows:

```shell
uvicorn app:app --reload
```

Access FastAPI UI by open your browser at http://127.0.0.1:8000/docs. Open the POST method tab and click on *Try it out*.
You can manipulate the JSON request body as you desire. Clicking on *Execute* does the following:
 
- Make an API call to [STATWORX COVID-19 API](https://github.com/STATWORX/covid-19-api) to fetch the latest data
- Check whether the desired model has already been trained and eventually trains it 
- Makes prediction in your desired prediction horizon

The default JSON request body:
```json
{
  "pred_start": "2020-11-01",
  "type": "auto_arima",
  "horizon": 7
}
```


## Currently implemented models

The following models are currently implemented:

- [AutoARIMA](https://www.sktime.org/en/latest/modules/auto_generated/sktime.forecasting.arima.AutoARIMA.html#sktime.forecasting.arima.AutoARIMA)
- [Prophet (from Facebook Open Source)](https://facebook.github.io/prophet/)





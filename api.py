import uvicorn
import orjson
import typing

from starlette.responses import JSONResponse
from src.train import *
from fastapi import FastAPI


# Define custom JSON library for handling nan
class ORJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content: typing.Any) -> bytes:
        return orjson.dumps(content)


api = FastAPI(default_response_class=ORJSONResponse)


# Expose the prediction functionality
@api.post('/predict')
def predict(covid: CovidInput):
    # Get parameters from API call
    data = covid.dict()

    # Instantiate model
    model = CovidModel(data['pred_start'], data['type'], data['horizon'])
    pred_dates = pd.date_range(start=data['pred_start'], periods=data['horizon']).to_period('D')

    forecasts = model.predict(pred_dates)

    # Prepare output
    date, target, prediction = model.prepare_output(forecasts, data['horizon'])

    return {'date': date,
            'target': target,
            'prediction': prediction}


if __name__ == '__main__':
    uvicorn.run(api, host='127.0.0.1', port=8000)

import uvicorn
import pandas as pd

from src.train import *
from fastapi import FastAPI

app = FastAPI()


# Expose the prediction functionality
@app.post('/predict')
def predict(covid: CovidInput):

    # Get parameters from API call
    data = covid.dict()

    # Instantiate model
    model = CovidModel(data['pred_start'], data['type'], data['horizon'])
    pred_dates = pd.date_range(start=data['pred_start'], periods=data['horizon']).to_period('D')

    # Make forecast
    forecast = model.predict(pred_dates, data['type']).to_list()

    return {'date': pred_dates.astype(str).tolist(),
            'forecast': forecast}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

import os
import fastapi
import pandas as pd

from challenge.model import DelayModel

app = fastapi.FastAPI()

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(request: fastapi.Request) -> dict:
    model = DelayModel()
    if not model.is_fitted:
        parpath = os.path.dirname(os.path.realpath(__file__))
        datapath = os.path.join(parpath, '../data/data.csv')
        data = pd.read_csv(datapath, low_memory=False)
        train_X, train_y  = model.preprocess(data, target_column='delay')
        model.fit(train_X, train_y)
    json_body = await request.json()
    try:
        X = model.preprocess(pd.DataFrame.from_records(json_body['flights']))
    except ValueError:
        raise fastapi.HTTPException(status_code=400, detail='Unknown column value in data')
    y_pred = model.predict(X)
    return {
        "predict": y_pred
    }
    
    
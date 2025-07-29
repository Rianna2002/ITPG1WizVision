from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import pandas as pd

app = FastAPI(root_path="/default/resaleprice-lambda-api")


# Load your model and data
xgb_model = xgb.Booster()
xgb_model.load_model("hdb_xgboost_model.model")
prev_prices = pd.read_csv("prev_prices.csv")

feature_names = [
    'flat_type', 'block', 'storey_range', 'floor_area_sqm', 'lease_commence_date', 'remaining_lease',
    'town_ANG MO KIO', 'town_BEDOK', 'town_BISHAN', 'town_BUKIT BATOK', 'town_BUKIT MERAH', 'town_BUKIT PANJANG',
    'town_BUKIT TIMAH', 'town_CENTRAL AREA', 'town_CHOA CHU KANG', 'town_CLEMENTI', 'town_GEYLANG', 'town_HOUGANG',
    'town_JURONG EAST', 'town_JURONG WEST', 'town_KALLANG/WHAMPOA', 'town_MARINE PARADE', 'town_PASIR RIS',
    'town_PUNGGOL', 'town_QUEENSTOWN', 'town_SEMBAWANG', 'town_SENGKANG', 'town_SERANGOON', 'town_TAMPINES',
    'town_TOA PAYOH', 'town_WOODLANDS', 'town_YISHUN',
    'flat_model_2-ROOM', 'flat_model_3GEN', 'flat_model_ADJOINED FLAT', 'flat_model_APARTMENT', 'flat_model_DBSS',
    'flat_model_IMPROVED', 'flat_model_IMPROVED-MAISONETTE', 'flat_model_MAISONETTE', 'flat_model_MODEL A',
    'flat_model_MODEL A-MAISONETTE', 'flat_model_MODEL A2', 'flat_model_MULTI GENERATION', 'flat_model_NEW GENERATION',
    'flat_model_PREMIUM APARTMENT', 'flat_model_PREMIUM APARTMENT LOFT', 'flat_model_PREMIUM MAISONETTE',
    'flat_model_SIMPLIFIED', 'flat_model_STANDARD', 'flat_model_TERRACE', 'flat_model_TYPE S1', 'flat_model_TYPE S2',
    'year', 'month_num', 'prev_month_resale_price'
]


# Define input format
class PredictionRequest(BaseModel):
    flat_type: int
    block: int
    storey_range: int
    floor_area: float
    lease_commence: int
    remaining_lease: int
    town: list[int]
    flat_model: list[int]
    year: int
    month: int

def get_prev_price(flat_type, year, month):
    row = prev_prices[(prev_prices['flat_type'] == flat_type) & 
                      (prev_prices['year'] == year) & 
                      (prev_prices['month_num'] == month - 1)]
    if not row.empty:
        return float(row['prev_month_resale_price'].values[0])
    fallback = prev_prices[(prev_prices['flat_type'] == flat_type) & 
                           ((prev_prices['year'] < year) | 
                           ((prev_prices['year'] == year) & (prev_prices['month_num'] < month - 1)))]
    if not fallback.empty:
        return float(fallback.sort_values(by=['year', 'month_num'], ascending=False)['prev_month_resale_price'].values[0])
    return np.nan

@app.post("/")
def predict(req: PredictionRequest):
    prev_price = get_prev_price(req.flat_type, req.year, req.month)
    
    features = np.array([
        req.flat_type, req.block, req.storey_range, req.floor_area, req.lease_commence, req.remaining_lease,
        *req.town, *req.flat_model, req.year, req.month, prev_price
    ]).reshape(1, -1)
    
    dmatrix = xgb.DMatrix(features, feature_names=feature_names)
    prediction = xgb_model.predict(dmatrix)[0]
    
    return {"predicted_price": float(round(prediction, 2))}

from mangum import Mangum

lambda_handler = Mangum(app)


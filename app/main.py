from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
import torch
import numpy as np



multi_class =torch.load('../models/pytorch_multi_classification_beer_v2.pt')

# read root
@app.get("/")
def read_root():
    return {"Hello": "World"}

# health check 
@app.get('/health', status_code=200)
def healthcheck():
    return 'Multi Classification is all ready to go!'

#format features
def format_features(brewery_name: str, review_aroma: float , review_appearance: float, review_palate: float, review_taste: float, beer_abv: float):
    return {
        'Brewery name': [brewery_name],
        'Review Aroma': [review_aroma],
        'Review Appearance': [review_appearance],
        'Review Palate': [review_palate],
	'Review Taste': [review_taste],
	'Beer Abv': [beer_abv]
    }


# # Prediction
# @app.get("/predict/beer")
# def predict_jm(brewery_name: str, review_aroma: int , review_appearance:int, review_palate: int, review_taste:int, beer_abv: int):
# # convert csv dataframe into series & get the index of the brewery_name
#     brew_index = list(brew_name['0'].squeeze())
#     brew_index = brew_index.index(brewery_name)
#     features = format_features(brewery_name, review_aroma, review_apperance, review_palate, review_taste, beer_abv)
#     obs = pd.DataFrame(features)
#     pred = multi_class.predict(obs)
#     return JSONResponse(pred.tolist())

brew_name = pd.read_csv('../data/brewery_name_list.csv')
beer_style = pd.read_csv('../data/beer_style_list.csv')

@app.get("/predict/beer")
def predict(brewery_name, review_aroma, review_appearance, review_palate, review_taste, beer_abv):

    # convert csv dataframe into series & get the index of the brewery_name
    brew_index = list(brew_name['0'].squeeze())
    brew_index = brew_index.index(brewery_name)

    # convert the features into a dict dataset
    features = format_features(brew_index, review_aroma, review_appearance, review_palate, review_taste, beer_abv)
    obs = pd.DataFrame(features)

    # convert observations to Tensor
    obs_dataset = torch.Tensor(np.array(obs))

    # Make predictions
    with torch.no_grad():
         output = multi_class(obs_dataset) 

    # convert output tensor to numpy - use astype to convert to integer
    output = torch.argmax(output).numpy().astype(int) 

    # final output
    return { 'Predicted beer style is =>' : beer_style.squeeze()[output] } 

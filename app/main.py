
from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
import torch
from src.models.pytorch import PytorchMultiClass
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

app = FastAPI()

class PytorchMultiClass(nn.Module):
    def __init__(self, num_features):
        super(PytorchMultiClass, self).__init__()
        
        self.layer_1 = nn.Linear(num_features, 32)
        self.layer_out = nn.Linear(32, 104)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.dropout(F.relu(self.layer_1(x)), training=self.training)
        x = self.layer_out(x)
        return self.softmax(x)

multi_class = PytorchMultiClass(6)

# multi_class = torch.load('../models/pytorch_multi_beer_style.pt')
multi_class.load_state_dict(torch.load('../models/pytorch_multi_classification_beer_v2.pt'))
# multi_class = torch.load('../pytorch_multi_classification_beer_250322.pt')

# model = MyModelDefinition(args)
# model.load_state_dict(torch.load(﻿'load/from/path/model.pth'﻿)﻿)

# read root
@app.get("/")
def read_root():
    return {"Hello": "World"}

# health check 
@app.get('/health', status_code=200)
def healthcheck():
    return 'Multi Classification is ready to go!'

#format features
def format_features(brew_index: int, review_aroma: float , review_appearance: float, review_palate: float, review_taste: float, beer_abv: float):
    return {
        'Brewery name': [brew_index],
        'Review Aroma': [review_aroma],
        'Review Appearance': [review_appearance],
        'Review Palate': [review_palate],
	'Review Taste': [review_taste],
	'Beer Abv': [beer_abv]
    }



brew_name = pd.read_csv('./data/brewery_name_list.csv')
beer_style = pd.read_csv('./data/beer_style_list.csv')

@app.get("/predict/beer")
def predict(brewery_name: str, review_aroma: float , review_appearance: float, review_palate: float, review_taste: float, beer_abv: float):

    # convert csv dataframe into series & get the index of the brewery_name
    brew_index = list(brew_name['0'].squeeze())
    brew_index = brew_index.index(brewery_name)

    # convert the features into a dict dataset
    features = format_features(brew_index, review_aroma, review_appearance, review_palate, review_taste, beer_abv)
    obs = pd.DataFrame(features)
    # print(obs)

    # convert observations to Tensor
    obs_dataset = torch.Tensor(np.array(obs))
    # print(obs_dataset)

    # # Make predictions
    with torch.no_grad():
         output = multi_class(obs_dataset)

    # print(output) 

    # # convert output tensor to numpy - use astype to convert to integer
    output = torch.argmax(output).numpy().astype(int) 

    # final output
    # return { 'Predicted beer style is =>' : 'beer_style.squeeze()[output]' } 
    return { 'Predicted beer style is =>' : beer_style.squeeze()[output] } 
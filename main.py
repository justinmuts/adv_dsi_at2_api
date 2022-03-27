
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

from typing import List, Optional
from fastapi import FastAPI, Query, Response

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

# READ ROOT AND GENERATE THE RESPONSE STATUS CODE
@app.get("/")
def read_root():
    return {"Hello": "World"}

# health check 
@app.get('/health', status_code=200)
def healthcheck():
    return 'Multi Classification is ready to go!'

# FORMAT FEATURES
def format_features(brewery_name: str, review_aroma: float , review_appearance: float, review_palate: float, review_taste: float, beer_abv: float):
    # convert csv dataframe into series & get the index of the brewery_name
    brew_index = list(brew_name['0'].squeeze())
    brew_index = brew_index.index(brewery_name)
    return {
        'Brewery name': [brew_index],
        'Review Aroma': [review_aroma],
        'Review Appearance': [review_appearance],
        'Review Palate': [review_palate],
	'Review Taste': [review_taste],
	'Beer Abv': [beer_abv]
    }


# THE BREWERY NAME LIST AND BEER STYLE LIST 
brew_name = pd.read_csv('./data/brewery_name_list.csv')
beer_style = pd.read_csv('./data/beer_style_list.csv')


# PREDICTION OF BEER (SINGLE)
@app.get("/beer/type/")
def predict(brewery_name: str, review_aroma: float , review_appearance: float, review_palate: float, review_taste: float, beer_abv: float):

    # convert the features into a dict dataset
    features = format_features(brewery_name, review_aroma, review_appearance, review_palate, review_taste, beer_abv)
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

    # FINAL OUTPUT
    # return { 'Predicted beer style is =>' : 'beer_style.squeeze()[output]' } 
    return { 'Predicted beer style is =>' : beer_style.squeeze()[output] } 

#--------------------
# IDENTIFY NUMBER OF RECORDS IN dict()
def dict_len(dict):
    no_count = sum([1 if isinstance(dict[x], (str, int))
                 else len(dict[x]) 
                 for x in dict])
    return no_count
#---------------------------------------------------
# RETURNING BEER PREDICTIONS FOR A MULTIPLE INPUTS 
#---------------------------------------------------
@app.get("/beers/type/")
def predict_beers(brew_input: Optional[List[str]] = Query(None)):
    query_items = {"input brewery_names":brew_input}
    #, review_aroma: float , review_appearance: float, review_palate: float, review_taste: float, beer_abv: float):
    
    # Create index for brewery name descriptions
    brew_index = list(brew_name['0'].squeeze())

    # setup dataframe for beer input & output
    df_beer = pd.DataFrame(columns=['brewery_name', 'review_aroma', 'review_appearance', 'review_palate', 'review_taste', 'beer_abv', 'brew_ind', 'beer_predict']) #'beer_style'])

    # extract features from dict input strings - separate input rows from dict
    # run predict() function on one dict{} beer-ratings

    for i in range(dict_len(query_items)):
        # get single string from fast api "GET"
        beer_val = [value[i] for value in query_items.values()]
        # separate one string into list of multiple string i.e. Features
        beer_str = ' '.join(beer_val).split(',')
        
        # create single row dict - convert features 1-5 into float 
        # - plus brew_ind for brewery name index
        dict = {'brewery_name': beer_str[0],  
                'review_aroma': float(beer_str[1]), 
                'review_appearance': float(beer_str[2]), 
                'review_palate': float(beer_str[3]), 
                'review_taste': float(beer_str[4]), 
                'beer_abv': float(beer_str[5]),
                'brew_ind': brew_index.index(beer_str[0])
        }
        # run predictions based on user input
        result = predict(dict['brewery_name'],dict['review_aroma'],dict['review_appearance'],
                              dict['review_palate'],dict['review_taste'],dict['beer_abv'])

        # print(beer_style.squeeze()[result])
        # add prediction to dict{}
        dict.update({
                "beer_predict": result 
                # "beer_style": beer_style.squeeze()[result]
                })
        
        # add dict{} to dataframe()
        df_beer = df_beer.append(dict, ignore_index = True)
        
    # return beef predictions   
    return df_beer['beer_predict'].to_dict()




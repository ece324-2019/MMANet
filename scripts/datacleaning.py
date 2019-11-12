"""
Code for data cleaning
Balanced Red and Blue Wins
"""

import numpy as np
import pandas as pd
from models import *

np.random.seed(420)
data = pd.read_csv("preprocessed_data.csv")
data = data.drop(columns=["B_Stance_Sideways", "title_bout"])
data_red = data[data["Winner"] == "Red"].copy()
data_blue = data[data["Winner"] == "Blue"].copy()

data_blue = data_blue.reset_index(drop=True)
data_red = data_red.reset_index(drop=True)
indices = list(range(len(data_red)))
np.random.shuffle(indices)
data_red_final = data_red[data_red.index.isin(indices[:1796])].copy()

data_red_to_blue = data_red[data_red.index.isin(indices[1796:])].copy()
data_red_to_blue.loc[:, "Winner"] = "Blue"
windata = data_red_to_blue.iloc[:, 0:2]
bluestats = data_red_to_blue.iloc[:, 2:68]
redstats = data_red_to_blue.iloc[:, 68:134]
blueage = data_red_to_blue.iloc[:, 134:135]
redage = data_red_to_blue.iloc[:, 135:136]
weights = data_red_to_blue.iloc[:, 136:150]
bluestance = data_red_to_blue.iloc[:, 150:154]
redstance = data_red_to_blue.iloc[:, 154:158]
data_red_to_bluefinal = windata.join(redstats).join(bluestats).join(redage).join(blueage).join(weights).join(
    redstance).join(bluestance)
data_red_to_bluefinal.columns = list(data_red_final.columns)
datafinal = pd.concat([data_blue, data_red_to_bluefinal, data_red_final], axis=0)
datafinal.loc[:, "Winner"] = (datafinal["Winner"] == "Red").astype(int)
datafinal = datafinal.sample(frac=1,random_state=420).reset_index(drop=True)
datafinal.to_csv("datafinal.csv", index=False)

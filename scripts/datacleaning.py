'''
Code for data cleaning
Balanced Red and Blue Wins
'''
import numpy as np
import pandas as pd
from models import *

np.random.seed(420)
data = pd.read_csv("preprocessed_data.csv")
data = data.drop(columns=["B_Stance_Sideways", "title_bout"])
datared = data[data["Winner"] == "Red"].copy()
datablue = data[data["Winner"] == "Blue"].copy()

datablue = datablue.reset_index(drop=True)
datared = datared.reset_index(drop=True)
indices = list(range(len(datared)))
np.random.shuffle(indices)
dataredfinal = datared[datared.index.isin(indices[:1796])].copy()

dataredtoblue = datared[datared.index.isin(indices[1796:])].copy()
dataredtoblue.loc[:, "Winner"] = "Blue"
windata = dataredtoblue.iloc[:, 0:2]
bluestats = dataredtoblue.iloc[:, 2:68]
redstats = dataredtoblue.iloc[:, 68:134]
blueage = dataredtoblue.iloc[:, 134:135]
redage = dataredtoblue.iloc[:, 135:136]
weights = dataredtoblue.iloc[:, 136:150]
bluestance = dataredtoblue.iloc[:, 150:154]
redstance = dataredtoblue.iloc[:, 154:158]
dataredtobluefinal = windata.join(redstats).join(bluestats).join(redage).join(blueage).join(weights).join(
    redstance).join(bluestance)
dataredtobluefinal.columns = list(dataredfinal.columns)
datafinal = pd.concat([datablue, dataredtobluefinal, dataredfinal], axis=0)
datafinal.loc[:, "Winner"] = (datafinal["Winner"] == "Red").astype(int)
datafinal = datafinal.sample(frac=1,random_state=420).reset_index(drop=True)
datafinal.to_csv("datafinal.csv", index=False)

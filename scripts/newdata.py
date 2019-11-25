"""
Code for data cleaning
Balanced Red and Blue Wins
"""

import numpy as np
import pandas as pd
from models import *

np.random.seed(420)
data = pd.read_csv("../data/data.csv")

data = data.fillna(0)
print(data["winner"].value_counts())
column_drop = ["B_HomeTown", "B_ID", "B_Location", "B_Name", "Date",
               "Event_ID", "Fight_ID", "Last_round", "R_HomeTown", "R_ID",
               "R_Location", "R_Name", "winby"]
data = data.drop(columns=column_drop)
data = data[data["winner"] != "no contest"]
data = data[data["winner"] != "draw"]

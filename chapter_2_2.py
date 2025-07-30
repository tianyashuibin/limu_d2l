import pandas as pd
import numpy as np
import torch

data = pd.read_csv("house_tiny.csv")
print( data)

inputs = data.iloc[:, 0:2]
outputs = data.iloc[:, 2]
print(inputs, outputs)

inputs["NumRooms"] = inputs["NumRooms"].fillna(inputs["NumRooms"].mean())

inputs = pd.get_dummies(inputs, dummy_na=True)
print( inputs)

inputs[["Alley_Pave", "Alley_nan"]] = inputs[["Alley_Pave", "Alley_nan"]].astype(int)
print(inputs)

x = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
print(x, y)
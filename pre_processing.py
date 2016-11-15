import pandas as pd
import numpy as np

print('Start reading file')
df = pd.read_csv('data/train.csv', delimiter=';', nrows = 10000)
print(df.shape())
print(df.head())

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

df = pd.read_csv('./data/weatherAUS.csv')
df['RISK_MM'] = df['Rainfall']
df2 = df[['RISK_MM', 'Location']]
X = pd.get_dummies(df2).values
X = StandardScaler().fit_transform(X)
Y = df['RainTomorrow'].values
Y = LabelEncoder().fit_transform(Y)
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.30, random_state=101)

print("hello")

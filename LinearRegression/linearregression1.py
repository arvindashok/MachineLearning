import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('lr_prices1.csv')

plt.scatter(df.area, df.price, color='blue', marker='+')
plt.xlabel('area')
plt.ylabel('price')

reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)

reg.predict([[3300]])
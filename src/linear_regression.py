import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/wine.csv")

x = df['alcohol'].values.reshape(-1,1)
y = df['quality'].values


x_b = np.hstack([np.ones_like(x), x])
beta, residuals, rank, s = np.linalg.lstsq(x_b, y, rcond=None)
y_pred = x_b @ beta
plt.scatter(x, y, label='Data points')
plt.plot(x, y_pred, color='red', label='Fitted line')
plt.xlabel('alcohol')
plt.ylabel('quality')
plt.legend()
plt.show()

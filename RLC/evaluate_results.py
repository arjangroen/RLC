import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch


filename = "end_result.csv"
df = pd.read_csv(filename)

# df['Return'] = df['Return'].apply(lambda x: float(eval('torch.' + x)))
df = df[df['iter'] < 190]
df.groupby('iter')['Return'].mean().plot()
plt.show()

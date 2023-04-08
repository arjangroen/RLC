import pandas as pd
import matplotlib.pyplot as plt


mean_returns = []
for i in range(14):
    istr = str((i+1)*10)
    filename = "end_result_" + istr
    df = pd.read_csv(filename)
    mean_returns.append(df['Return'].mean())

ema = []
ema_current = 0
inverse_gamma = 1
i = 1

for r in mean_returns:
    gamma = 1 / min(i, inverse_gamma)
    ema_current = (1-gamma) * ema_current + gamma * r
    ema.append(ema_current)
    i += 1

plt.plot(ema)
plt.show()

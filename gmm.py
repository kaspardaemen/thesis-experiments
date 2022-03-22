import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

def sin (x, a, h , b):
    return a * np.sin((x-h)/b)


def load_dataframe(day, year, ss) -> pd.DataFrame:
    """ Create a time series x sin wave dataframe. """
    sin_columns = [f'sin{i}' for i in range(ss)]
    df = pd.DataFrame(columns=['date'] + sin_columns)
    df.date = pd.date_range(start='2018-01-01', end='2021-03-01', freq='D')
    a = np.random.normal(loc=1, scale=0.1, size=ss)
    h = np.random.normal(loc=0, scale=0.1, size=ss)
    b = np.random.normal(loc=1, scale=0.1, size=ss)
    for i, column in enumerate(sin_columns):
        df.loc[:, column] = 1 + sin(df.date.astype('int64') // 1e9 * (2 * np.pi / year), a[i], h[i], b[i])
        df.loc[:, column] = (df.loc[:, column] * 100).round(2)
    df.date = df.date.apply(lambda d: d.strftime('%Y-%m-%d'))
    df.date = pd.to_datetime(df.date)
    return df.set_index('date')

def learn_and_sample(n_components, n_samples):
  plt.figure()
  gmm = mixture.GaussianMixture(n_components=n_components).fit(X)
  S, labels = gmm.sample(n_samples)
  for i in range(5):
      plt.plot(S[i])
  plt.title(f'{n_samples} generated samples with n_components = {n_components}')
  plt.show()
  
# CREATE AND PLOT ORIGINAL DISTRIBUTION
day = 24 * 60 * 60
year = 365.2425 * day
ss = 1000

df = load_dataframe(day, year, ss)
df[[f'sin{i}' for i in range(30,36)]].plot()
plt.title('5 samples from the real distribution')
plt.show()

# SAMPLE AND PLOT LEARNED DISTRIBUTION

X = df.transpose().to_numpy()
from sklearn import mixture
# n_components = np.arange(1,17)
# models = [mixture.GaussianMixture(n_components=i).fit(X) for i in n_components]
#
# plt.plot(n_components, [m.bic(X) for m in models], label='BIC')
# plt.plot(n_components, [m.aic(X) for m in models], label='AIC')
# plt.legend(loc='best')
# plt.xlabel('n_components');
# plt.show()

learn_and_sample(n_components=1, n_samples=5)

learn_and_sample(n_components=5, n_samples=5)
    
learn_and_sample(n_components=10, n_samples=5)    
    








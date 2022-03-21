# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
sns.set()

def sin (x, a, h , b):
    return a * np.sin((x-h)/b)


def load_dataframe(day, year, ss) -> pd.DataFrame:
    """ Create a time series x sin wave dataframe. """
    sin_columns = [f'sin{i}' for i in range(ss)]
    df = pd.DataFrame(columns=['date'] + sin_columns)
    df.date = pd.date_range(start='2018-01-01', end='2021-03-01', freq='D')
    a = stats.norm(loc=1, scale=0.1).rvs(ss)
    h = stats.norm(loc=0, scale=0.1).rvs(ss)
    b = stats.norm(loc=1, scale=0.1).rvs(ss)
    for i, column in enumerate(sin_columns):
        df.loc[:, column] = 1 + sin(df.date.astype('int64') // 1e9 * (2 * np.pi / year), a[i], h[i], b[i])
        df.loc[:, column] = (df.loc[:, column] * 100).round(2)
    df.date = df.date.apply(lambda d: d.strftime('%Y-%m-%d'))
    df.date = pd.to_datetime(df.date)
    return df.set_index('date')



if __name__ == '__main__':
    day = 24 * 60 * 60
    year = 365.2425 * day
    ss = 1000

    df = load_dataframe(day, year, ss)
    df[[f'sin{i}' for i in range(5)]].plot()
    plt.show()
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

    gmm = mixture.GaussianMixture(n_components=10).fit(X)
    S, labels = gmm.sample(5)
    for i in range(5):
        plt.plot(S[i])
    plt.show()









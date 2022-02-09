from exmples.Regression.LinearRegression import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['figure.figsize'] = [8, 5]
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'
plt.style.use('seaborn-whitegrid')

options = {
    "Linear_regression": [LinearRegression(), True]
}

for val in options.values():
    if val[1]:
        df = pd.read_csv('Resources/insurance.csv')
        categorical_columns = ['sex', 'children', 'smoker', 'region']
        df_encode = pd.get_dummies(data=df, prefix='OHE', prefix_sep='_',
                                   columns=categorical_columns,
                                   drop_first=True,
                                   dtype='int8')
        val[0].start()

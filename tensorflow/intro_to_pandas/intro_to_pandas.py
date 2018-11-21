from __future__ import print_function
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

pd.__version__

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])

population =  pd.Series([852469,1015785,485199])

cities = pd.DataFrame({"city name":city_names,"population":population})

# california_housing_dataFrame = pd.read_csv("https://download.mlcc.google.cn/mledu-datasets/california_housing_train.csv", sep=",")
#
# california_housing_dataFrame.describe()
# # print(california_housing_dataFrame)
#
#
# california_housing_dataFrame.head()
#
# california_housing_dataFrame.hist('housing_median_age')
# plt.show()
#


# print(cities['city name'])
#
# print(cities[0:2])

population / 1000

np.log(population)

population.apply(lambda val:val > 10000000)

cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['population'] / cities['Area square miles']

# print(cities)

cities['is wide and has saint name'] = (cities['Area square miles']>50)&cities['city name'].apply(lambda name:name.startswith("San"))

print(cities)

print(cities.index)

cities.reindex([0,4,5,2])
print(cities)
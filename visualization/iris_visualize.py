
# coding: utf-8

# In[39]:

# import
import scipy
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.tools.plotting import scatter_matrix
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D


# In[40]:

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data'
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal width', 'class']
dataframe = pd.read_csv(url, names=names)
data = dataframe.values
# target = pd.Series(data[:,4:5])


# In[41]:

def create_color(i):
    if i == 'Iris-setosa':
        return 1
    elif i == 'Iris-versicolor':
        return 2
    else:
        return 3


# In[42]:

target = data[:,4:5]
color = [create_color(i) for i in target]


# In[43]:

data_input = data[:,0:4]

# TSNE fit
model = TSNE(n_components=3, random_state=0)
array = model.fit_transform(data_input)
spl = array[:,0:1]
spw = array[:,1:2]
sz = array[:,2:3]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(spl, spw, sz, c = color)
plt.show()


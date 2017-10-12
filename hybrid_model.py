# -*- coding: utf-8 -*-

# Part 1 - Identify the Frauds with the Self-Organizing Map

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Credit_Card_Applications.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(3,8)], mappings[(5,7)]), axis = 0)
frauds = sc.inverse_transform(frauds)

# Part 2 - Going from Unsupervised to Supervised Deep Learning

# We need a dependent variable to go from unsupervised to supervised, as 
# unsupervised we don't need any dependent variable because it was trained
# on our features without using any dependent variables

# Supervised needs dependent variables because we need the model to understand 
# correlations between features and results, outcomes is the dependent

# Combine the two models, take input from results of SOMs
# Ranking of predicted probablity of each customers cheated


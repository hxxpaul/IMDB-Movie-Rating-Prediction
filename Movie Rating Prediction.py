#!/usr/bin/env python
# coding: utf-8

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


raw = pd.read_csv('movie_metadata.csv')
raw.head(2)
raw.shape

# check missing value
raw.isnull().sum()

# the remaining dataset is still in good shape and has enough instances
# besides, some categorical values are not estimatable
# so drop instances with missing value is feasible

df = raw.reindex(sorted(raw), axis =1).dropna().drop_duplicates() # drop duplicates

# convert inproper data types into appropriate ones
df['title_year'] = df['title_year'].astype(int)

# replace genres with prime genre
genre = df.genres.str.split('|').tolist()
prime_genre = [i[0] for i in genre]
# drop trivial variables in this project
df = df.drop(['genres','movie_imdb_link','movie_title', 'plot_keywords'], axis = 1)
df['prime_genre'] = prime_genre

# seperate catogorical and numerical variables
dtype_pairs = dict(df.dtypes).items()
obj_col = [key for key, value in dtype_pairs if value == 'object']
num_col = list(np.setdiff1d(df.columns, obj_col))
# seperate independent and dependent variables
X_col = [i for i in num_col if i != 'imdb_score']

# univariate outlier
fig, axs = plt.subplots(3, 1, figsize = (10,12))
sns.boxplot(ax = axs[0], x = df['actor_1_facebook_likes'])
axs[0].set_title('Actor 1 Facebook Likes')
sns.boxplot(ax = axs[1], x = df['actor_2_facebook_likes'])
axs[1].set_title('Actor 2 Facebook Likes')
sns.boxplot(ax = axs[2], x = df['actor_3_facebook_likes'])
axs[2].set_title('Actor 3 Facebook Likes')
fig.tight_layout();

# multivariate outlier
fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(df['budget'], df['duration'])
ax.set_xlabel('budget')
ax.set_ylabel('duration');

# replace outliers with the threshold value of IQR
def remove_outlier(col):
    sorted(col)
    Q1, Q3 = col.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range, upper_range

for col in X_col:
    lower_cap, upper_cap = remove_outlier(df[col])
    df[col] = np.where(df[col] > upper_cap, upper_cap, df[col])
    df[col] = np.where(df[col] < lower_cap, lower_cap, df[col])

# boxplot after removing outlier
fig, axs = plt.subplots(3, 1, figsize = (12,12))
sns.boxplot(ax = axs[0], x = df['actor_1_facebook_likes'])
axs[0].set_title('Actor 1 Facebook Likes')
sns.boxplot(ax = axs[1], x = df['actor_2_facebook_likes'])
axs[1].set_title('Actor 2 Facebook Likes')
sns.boxplot(ax = axs[2], x = df['actor_3_facebook_likes'])
axs[2].set_title('Actor 3 Facebook Likes')
fig.tight_layout();

# scatter plot after removing outlier
fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(df['budget'], df['duration'])
ax.set_xlabel('budget')
ax.set_ylabel('duration');

# correlation matrix and heatmap

# since the primary goal of this project is to predict, not interpret the coefficients
# which are severely influenced by multicollinearity, it is okay to leave those
# variables with high correlation
corr = df.corr()
corr.head()
sns.heatmap(corr, cmap = 'Reds')

# prime genre in barplot
genres = df['prime_genre'].value_counts(normalize = True)
genres.plot.barh(figsize = (6, 8));

# country in pie chart
fig, axs = plt.subplots(1, 2, figsize = (12,12))
color = df['color'].value_counts()
color.plot.pie(ax = axs[0])
country = df['country'].value_counts()
country.plot.pie(ax = axs[1])
fig.tight_layout();

df['log_budget'] = np.log10(df['budget'])
df['log_gross'] = np.log10(df['gross'])

# pair plot
sns.pairplot(df, 
             vars = ['log_budget', 'log_gross', 'imdb_score'], 
             hue = 'color', diag_kind = 'kde', 
             plot_kws = {'alpha': 0.3, 's': 30, 'edgecolor': 'k'},
             height = 4);

# in order to prevent the future modeling being dominated by certain large-scale variables,
# it is necessary to perform standardization
df.describe().transpose()
# standardize only the numerical independent variables
df = df.drop(['log_budget', 'log_gross'], axis = 1)
df[X_col] = StandardScaler().fit_transform(df[X_col])
df.describe().transpose()

# labeling categorical variables
# unique values in each categorical variables
obj_dict = {obj_col[i] : len(df[obj_col[i]].unique()) for i in range(len(obj_col))}
obj_dict

# considering the large unique value counts in categorical variables
# it is more reasonable to use label encoding instead of one-hot encoding
for col in obj_col:
    df[col] = LabelEncoder().fit_transform(df[col])

# trainset and testset
df_train = df.sample(frac = 0.8, random_state = 1)
df_test = df.drop(df_train.index)

# independent and dependent variables
train_x = df_train.copy()
test_x = df_test.copy()
train_y = df_train.pop('imdb_score')
test_y = df_test.pop('imdb_score')

# build normalization layer
normalizer = preprocessing.Normalization()

# linear model
# build and compile model in sequence
linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units = 1)])

linear_model.compile(
    optimizer = tf.optimizers.Adam(learning_rate = 0.1),
    loss = 'mean_absolute_error')

# fit model
history = linear_model.fit(
    train_x, train_y,
    epochs = 100, verbose = 0,
    validation_split = 0.2)

# visualization of training process
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [imdb_score]')
    plt.legend()
    plt.grid(True)

plot_loss(history)

# evaluate model
test_results = {}
test_results['linear_model'] = linear_model.evaluate(
    test_x, test_y, verbose=0)

# DNN model applying rectified linear unit activation function and Adam algorithm optimizer
# add neuron layers besides normalization layer
def model_1(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
    return model

dnn_model_1 = model_1(normalizer)

history = dnn_model_1.fit(
    train_x, train_y,
    verbose=0, epochs=100,
    validation_split=0.2)

plot_loss(history)

test_results['dnn_model_1'] = dnn_model_1.evaluate(test_x, test_y, verbose=0)

# DNN model applying Hyperbolic tangent activation function and Adamax algorithm optimizer
def model_2(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='tanh'),
        layers.Dense(64, activation='tanh'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adamax(0.001))
    return model

dnn_model_2 = model_2(normalizer)

history = dnn_model_2.fit(
    train_x, train_y,
    verbose=0, epochs=100,
    validation_split=0.2)

plot_loss(history)

test_results['dnn_model_2'] = dnn_model_2.evaluate(test_x, test_y, verbose=0)

# performance comparison
pd.DataFrame(test_results, index = ['Mean absolute error (imdb_score)']).transpose()
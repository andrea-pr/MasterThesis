import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import backend as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dropout, Dense, Activation, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import sklearn
from keras_preprocessing.image import ImageDataGenerator
import tensorflow.keras as keras 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.inception_v3 import InceptionV3
from sklearn.utils import resample
import scipy.stats as st
from tensorflow.keras.models import load_model
from itertools import chain
import argparse
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from functions import r_squared

parser = argparse.ArgumentParser()
parser.add_argument("--variable", type=str, default='Age')
parser.add_argument("--save_name", type=str, default='age')
parser.add_argument("--fig_name", type=str, default='Age')
parser.add_argument("--unit", type=str, default='years')
parser.add_argument("--path_outputs", type=str, default='/home/andreap/fundusdata_2/results_retina/')

args = parser.parse_args()
variable = args.variable
save_name = args.save_name
fig_name = args.fig_name
unit = args.unit
path_outputs = args.path_outputs

filepath = path_outputs + save_name + '_singleinput.hdf5'

#Load dataframe
test_df=pd.read_csv('/home/andreap/fundusdata_2/GoodQualityImages/Test/test_df.csv')
test_df_filtered = test_df[test_df[variable].notna()]

#Load model
filepath = path_outputs + save_name + '_singleinput.hdf5'
model = load_model(filepath)


#Sort test df
test_df_filtered = test_df_filtered.sort_values(['eid', 'Image_ID'])

test_datagen=ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_v3.preprocess_input)

test_generator=test_datagen.flow_from_dataframe(
    dataframe=test_df_filtered,
    directory= '/home/andreap/fundusdata_2/GoodQualityImages/Test/',
    x_col="Image_ID",
    y_col=None,
    batch_size=1,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(587,587))

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

test_generator.reset()
y_predictions=model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)


test_df_filtered['predictions'] = y_predictions
left_right = [x[8:13] for x in list(test_df_filtered['Image_ID'])]
test_df_filtered['left_right'] = left_right



#All Images Predictions
print(variable, ' R2 All Images: ', r_squared(test_df_filtered[variable], test_df_filtered['predictions']))
print(variable, ' Mean Absolute Error(MAE) All Images: ', sklearn.metrics.mean_absolute_error(test_df_filtered[variable], test_df_filtered['predictions']))

#Left Eye Images Predictions
df_left = test_df_filtered[test_df_filtered['left_right'] == '21015']
print(variable, ' R2 Left Images: ', r_squared(df_left[variable], df_left['predictions']))
print(variable, ' Mean Absolute Error(MAE) Left Images: ', sklearn.metrics.mean_absolute_error(df_left[variable], df_left['predictions']))

#Right Eye Images Predictions
df_right = test_df_filtered[test_df_filtered['left_right'] == '21016']
print(variable, ' R2 Left Images: ', r_squared(df_right[variable], df_right['predictions']))
print(variable, ' Mean Absolute Error(MAE) Right Images: ', sklearn.metrics.mean_absolute_error(df_right[variable], df_right['predictions']))

mean_df = test_df_filtered.groupby('eid').mean()
print(variable, ' R2 Images Mean: ', r_squared(mean_df[variable], mean_df['predictions']))
print(variable, ' Mean Absolute Error(MAE) Images Mean: ', sklearn.metrics.mean_absolute_error(mean_df[variable], mean_df['predictions']))

test_df_filtered[['eid', 'Image_ID', variable, 'predictions']].to_csv(save_name + 'predictions.csv')
mean_df[['individual_id', variable, 'predictions']].to_csv(save_name + 'predictions_mean.csv')


lineStart = mean_df[variable].min() 
lineEnd = mean_df[variable].max()  

fig, ax = plt.subplots(figsize=(12, 12))
plt.scatter(mean_df[variable], mean_df['predictions'], s=4)
plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-')
plt.xlabel('True ' + fig_name + '(' + unit + ')', fontsize=16)
plt.ylabel('Predicted ' + fig_name + '(' + unit + ')', fontsize=16)
plt.xlim(lineStart, lineEnd)
plt.ylim(lineStart, lineEnd)
plt.tick_params(labelsize=16)
plt.show()
plt.savefig(fig_name + '_plot.png')

eid_ethnic = test_df_filtered[['eid', 'Ethnic background_0.0']].drop_duplicates()
mean_df = pd.merge(mean_df, eid_ethnic, left_on = 'individual_id', right_on = 'eid')

british_irish = mean_df[(mean_df['Ethnic background_0.0'] == 'British')|(mean_df['Ethnic background_0.0'] == 'Irish')]
eid = british_irish['eid']
other_than_british = mean_df[~mean_df.eid.isin(eid)]

print(variable, 'R2 British/Irish:', r_squared(british_irish[variable], british_irish['predictions']))
print(variable, 'MAE British/Irish:', sklearn.metrics.mean_absolute_error(british_irish[variable], british_irish['predictions']))

sample_size = len(test_df_filtered)

r2_samples_british = []

for i in range(0, 2000): 
    if i == 100: 
        print('100')
    if i == 500: 
        print('500')
    if i == 800: 
        print('800')
    predictions = resample(british_irish['predictions'], replace=True, n_samples=sample_size, random_state=i)
    true_y = resample(british_irish[variable], replace=True, n_samples=sample_size, random_state=i)
    r2_samples_british.append(r_squared(true_y, predictions))
    
print('R2 British', np.percentile(r2_samples_british, [2.5, 97.5]))

print('R2 Other than British/Irish:', r_squared(other_than_british[variable], other_than_british['predictions']))
print('MAE Other than British/Irish:', sklearn.metrics.mean_absolute_error(other_than_british[variable], other_than_british['predictions']))

r2_samples_not_british = []

for i in range(0, 2000): 
    if i == 100: 
        print('100')
    if i == 500: 
        print('500')
    if i == 800: 
        print('800')
    predictions = resample(other_than_british['predictions'], replace=True, n_samples=sample_size, random_state=i)
    true_y = resample(other_than_british[variable], replace=True, n_samples=sample_size, random_state=i)
    r2_samples_not_british.append(r_squared(true_y, predictions))
    
print('R2 Not British', np.percentile(r2_samples_not_british, [2.5, 97.5]))


eid_gender = test_df_filtered[['eid', 'Genetic sex_0.0']].drop_duplicates()
mean_df = pd.merge(mean_df, eid_gender, left_on = 'individual_id', right_on = 'eid')
male_df = mean_df[mean_df['Genetic sex_0.0'] == 'Male']
female_df = mean_df[mean_df['Genetic sex_0.0'] == 'Female']

print(variable, 'R2 Male:', r_squared(male_df[variable], male_df['predictions']))
print(variable, 'MAE Male:', sklearn.metrics.mean_absolute_error(male_df[variable], male_df['predictions']))

sample_size = len(test_df_filtered)

print('Non-parametric bootstrapping - All Images')

r2_samples_male = []

for i in range(0, 2000): 
    if i == 100: 
        print('100')
    if i == 500: 
        print('500')
    if i == 800: 
        print('800')
    predictions = resample(male_df['predictions'], replace=True, n_samples=sample_size, random_state=i)
    true_y = resample(male_df[variable], replace=True, n_samples=sample_size, random_state=i)
    r2_samples_male.append(r_squared(true_y, predictions))
    
print('R2 Male', np.percentile(r2_samples_male, [2.5, 97.5]))

print(variable, 'R2 Female:', r_squared(female_df[variable], female_df['predictions']))
print(variable, 'MAE Female:', sklearn.metrics.mean_absolute_error(female_df[variable], female_df['predictions']))

r2_samples_female = []

for i in range(0, 2000): 
    if i == 100: 
        print('100')
    if i == 500: 
        print('500')
    if i == 800: 
        print('800')
    predictions = resample(female_df['predictions'], replace=True, n_samples=sample_size, random_state=i)
    true_y = resample(female_df[variable], replace=True, n_samples=sample_size, random_state=i)
    r2_samples_female.append(r_squared(true_y, predictions))
    
print('R2 Female', np.percentile(r2_samples_female, [2.5, 97.5]))

gender_filtered = mean_df[mean_df['Genetic sex_0.0'].notna()]
colors = {'Female':'tab:blue', 'Male':'tab:orange'}
fig, ax = plt.subplots(figsize=(12, 12))
grouped = gender_filtered.groupby('Genetic sex_0.0')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x=variable, y='predictions', label=key, color=colors[key], s=4)
plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-')
plt.xlabel('True ' + fig_name + '(' + unit + ')', fontsize=16)
plt.ylabel('Predicted ' + fig_name + '(' + unit + ')', fontsize=16)
plt.xlim(lineStart, lineEnd)
plt.ylim(lineStart, lineEnd)
plt.tick_params(labelsize=16)
plt.legend(fontsize=16,markerscale=2)
plt.show()
plt.savefig(fig_name + 'gender_plot.png')


age_between39_50 = mean_df[(mean_df['Age'] > 39) & (mean_df['Age'] <= 50)]
print('Mean age', age_between39_50['Age'].mean())
age_above_50 = mean_df[mean_df['Age'] > 50]
print('Mean age', age_above_50['Age'].mean())
print(age_between39_50.head())


print(variable, 'R2 Age Between 39 and 50:', r_squared(age_between39_50[variable], age_between39_50['predictions']))
print(variable, 'MAE Age Between 39 and 50:', sklearn.metrics.mean_absolute_error(age_between39_50[variable], age_between39_50['predictions']))

r2_samples_between39_50 = []

for i in range(0, 2000): 
    if i == 100: 
        print('100')
    if i == 500: 
        print('500')
    if i == 800: 
        print('800')
    predictions = resample(age_between39_50['predictions'], replace=True, n_samples=sample_size, random_state=i)
    true_y = resample(age_between39_50[variable], replace=True, n_samples=sample_size, random_state=i)
    r2_samples_between39_50.append(r_squared(true_y, predictions))
    
print('R2 Between 39 and 50', np.percentile(r2_samples_between39_50, [2.5, 97.5]))

print(variable, 'R2 Age Above 50:', r_squared(age_above_50[variable], age_above_50['predictions']))

r2_samples_above_50 = []

for i in range(0, 2000): 
    if i == 100: 
        print('100')
    if i == 500: 
        print('500')
    if i == 800: 
        print('800')
    predictions = resample(age_above_50['predictions'], replace=True, n_samples=sample_size, random_state=i)
    true_y = resample(age_above_50[variable], replace=True, n_samples=sample_size, random_state=i)
    r2_samples_above_50.append(r_squared(true_y, predictions))
    
print('R2 Above 50', np.percentile(r2_samples_above_50, [2.5, 97.5]))

print(variable, 'MAE Age Above 50:', sklearn.metrics.mean_absolute_error(age_above_50[variable], age_above_50['predictions']))


#Bootstrapping sample Size -> Same Size as test set
sample_size = len(test_df_filtered)

print('Non-parametric bootstrapping - All Images')

mae_samples = []
r2_samples = []


for i in range(0, 2000): 
    if i == 100: 
        print('100')
    if i == 500: 
        print('500')
    if i == 800: 
        print('800')
    predictions = resample(test_df_filtered['predictions'], replace=True, n_samples=sample_size, random_state=i)
    true_y = resample(test_df_filtered[variable], replace=True, n_samples=sample_size, random_state=i)
    r2_samples.append(r_squared(true_y, predictions))
    mae_samples.append(sklearn.metrics.mean_absolute_error(true_y, predictions))
    
print('MAE All Images', np.percentile(mae_samples, [2.5, 97.5]))
print('R2 All Images', np.percentile(r2_samples, [2.5, 97.5]))



print('Non-parametric bootstrapping - Left Images')

mae_samples = []
r2_samples = []

for i in range(0, 2000): 
    if i == 100: 
        print('100')
    if i == 500: 
        print('500')
    if i == 800: 
        print('800')
    predictions = resample(df_left['predictions'], replace=True, n_samples=sample_size, random_state=i)
    true_y = resample(df_left[variable], replace=True, n_samples=sample_size, random_state=i)
    r2_samples.append(r_squared(true_y, predictions))
    mae_samples.append(sklearn.metrics.mean_absolute_error(true_y, predictions))
    
print('MAE Left Images', np.percentile(mae_samples, [2.5, 97.5]))
print('R2 Left Images', np.percentile(r2_samples, [2.5, 97.5]))
    

print('Non-parametric bootstrapping - Right Images')

mae_samples = []
r2_samples = []

for i in range(0, 2000): 
    if i == 100: 
        print('100')
    if i == 500: 
        print('500')
    if i == 800: 
        print('800')
    predictions = resample(df_right['predictions'], replace=True, n_samples=sample_size, random_state=i)
    true_y = resample(df_right[variable], replace=True, n_samples=sample_size, random_state=i)
    r2_samples.append(r_squared(true_y, predictions))
    mae_samples.append(sklearn.metrics.mean_absolute_error(true_y, predictions))
    
print('MAE Right Images', np.percentile(mae_samples, [2.5, 97.5]))
print('R2 Right Images', np.percentile(r2_samples, [2.5, 97.5]))



mae_samples = []
r2_samples = []

for i in range(0, 2000): 
    if i == 100: 
        print('100')
    if i == 500: 
        print('500')
    if i == 800: 
        print('800')
    predictions = resample(mean_df['predictions'], replace=True, n_samples=sample_size, random_state=i)
    true_y = resample(mean_df[variable], replace=True, n_samples=sample_size, random_state=i)
    r2_samples.append(r_squared(true_y, predictions))
    mae_samples.append(sklearn.metrics.mean_absolute_error(true_y, predictions))
    
print('MAE Mean Images', np.percentile(mae_samples, [2.5, 97.5]))
print('R2 Mean Images', np.percentile(r2_samples, [2.5, 97.5]))
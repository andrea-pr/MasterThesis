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
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc



#Load dataframe
test_df=pd.read_csv('/home/andreap/fundusdata_2/GoodQualityImages/Test/test_df.csv')
test_df_filtered = test_df[test_df['Genetic sex_0.0'].notna()]


#Load Model 
filepath = '/home/andreap/fundusdata_2/results_retina_alllayers/gender_singleinput.hdf5'
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
test_df_filtered['predictions_classes'] = [1 if x > 0.5 else 0 for x in y_predictions]
test_df_filtered['Genetic sex_0.0_integers'] = test_df_filtered['Genetic sex_0.0'].map({'Female': 0, 'Male': 1})

#Majority class predictions -> Female
test_df_filtered['naive_predictions'] = 0

#All Images Predictions
print('AUC All Images:', roc_auc_score(test_df_filtered['Genetic sex_0.0_integers'], test_df_filtered['predictions']))
precision, recall, _ = precision_recall_curve(test_df_filtered['Genetic sex_0.0_integers'], test_df_filtered['predictions'])
auc_score = auc(recall, precision)
print('AUROC All Images:', auc_score)

#Left Eye Images Predictions
df_left = test_df_filtered[test_df_filtered['left_right'] == '21015']
print('AUC Left Images:', roc_auc_score(df_left['Genetic sex_0.0_integers'], df_left['predictions']))
precision, recall, _ = precision_recall_curve(df_left['Genetic sex_0.0_integers'], df_left['predictions'])
auc_score = auc(recall, precision)
print('AUROC Left Images:', auc_score)

#Right Eye Images Predictions
df_right = test_df_filtered[test_df_filtered['left_right'] == '21016']
print('AUC Right Images:', roc_auc_score(df_right['Genetic sex_0.0_integers'], df_right['predictions']))
precision, recall, _ = precision_recall_curve(df_right['Genetic sex_0.0_integers'], df_right['predictions'])
auc_score = auc(recall, precision)
print('AUROC Right Images:', auc_score)

mean_df = test_df_filtered.groupby('eid').mean()
mean_df['predictions_classes'] = np.where(mean_df.predictions > 0.5, 1, 0)
print('AUC Mean Images:', roc_auc_score(mean_df['Genetic sex_0.0_integers'], mean_df['predictions']))
precision, recall, _ = precision_recall_curve(mean_df['Genetic sex_0.0_integers'], mean_df['predictions'])
auc_score = auc(recall, precision)
print('AUROC Mean Images:', auc_score)

eid_ethnic = test_df_filtered[['eid', 'Ethnic background_0.0']].drop_duplicates()
mean_df = pd.merge(mean_df, eid_ethnic, left_on = 'individual_id', right_on = 'eid')

british_irish = mean_df[(mean_df['Ethnic background_0.0'] == 'British')|(mean_df['Ethnic background_0.0'] == 'Irish')]
eid = british_irish['eid']
other_than_british = mean_df[~mean_df.eid.isin(eid)]


print('AUC British:', roc_auc_score(british_irish['Genetic sex_0.0_integers'], british_irish['predictions']))

sample_size = len(test_df_filtered)

auc_score_samples_british = []

for i in range(0, 2000): 
    if i == 100: 
        print('100')
    if i == 500: 
        print('500')
    if i == 800: 
        print('800')
    predictions_classes = resample(british_irish['predictions_classes'], replace=True, n_samples=sample_size, random_state=i)
    predictions = resample(british_irish['predictions'], replace=True, n_samples=sample_size, random_state=i)
    true_y = resample(british_irish['Genetic sex_0.0_integers'], replace=True, n_samples=sample_size, random_state=i)
    auc_score_samples_british.append(roc_auc_score(true_y, predictions))
print('AUC British', np.percentile(auc_score_samples_british, [2.5, 97.5]))

print('AUC Not British:', roc_auc_score(other_than_british['Genetic sex_0.0_integers'], other_than_british['predictions']))

auc_score_samples_not_british = []

for i in range(0, 2000): 
    if i == 100: 
        print('100')
    if i == 500: 
        print('500')
    if i == 800: 
        print('800')
    predictions_classes = resample(other_than_british['predictions_classes'], replace=True, n_samples=sample_size, random_state=i)
    predictions = resample(other_than_british['predictions'], replace=True, n_samples=sample_size, random_state=i)
    true_y = resample(other_than_british['Genetic sex_0.0_integers'], replace=True, n_samples=sample_size, random_state=i)
    auc_score_samples_not_british.append(roc_auc_score(true_y, predictions))
print('AUC Not British', np.percentile(auc_score_samples_not_british, [2.5, 97.5]))

age_between39_50 = mean_df[(mean_df['Age'] > 39) & (mean_df['Age'] <= 50)]
age_above_50 = mean_df[mean_df['Age'] > 50]


print('AUC new:', roc_auc_score(age_between39_50['Genetic sex_0.0_integers'], age_between39_50['predictions']))


sample_size = len(test_df_filtered)

auc_score_samples_age = []

for i in range(0, 2000): 
    if i == 100: 
        print('100')
    if i == 500: 
        print('500')
    if i == 800: 
        print('800')
    predictions_classes = resample(age_between39_50['predictions_classes'], replace=True, n_samples=sample_size, random_state=i)
    predictions = resample(age_between39_50['predictions'], replace=True, n_samples=sample_size, random_state=i)
    true_y = resample(age_between39_50['Genetic sex_0.0_integers'], replace=True, n_samples=sample_size, random_state=i)
    auc_score_samples_age.append(roc_auc_score(true_y, predictions))

    
print('AUC Age between 39 and 50', np.percentile(auc_score_samples_age, [2.5, 97.5]))

print('AUC new:', roc_auc_score(age_above_50['Genetic sex_0.0_integers'], age_above_50['predictions']))

sample_size_age = len(age_above_50)

auc_score_samples_age_2 = []

for i in range(0, 2000): 
    if i == 100: 
        print('100')
    if i == 500: 
        print('500')
    if i == 800: 
        print('800')
    predictions_classes = resample(age_above_50['predictions_classes'], replace=True, n_samples=sample_size, random_state=i)
    predictions = resample(age_above_50['predictions'], replace=True, n_samples=sample_size, random_state=i)
    true_y = resample(age_above_50['Genetic sex_0.0_integers'], replace=True, n_samples=sample_size, random_state=i)
    auc_score_samples_age_2.append(roc_auc_score(true_y, predictions))

    
print('AUC Age above 50', np.percentile(auc_score_samples_age_2, [2.5, 97.5]))


sample_size = len(test_df_filtered)

print('Non-parametric bootstrapping - All Images')

auc_score_samples = []
prc_score_samples = []

for i in range(0, 2000): 
    if i == 100: 
        print('100')
    if i == 500: 
        print('500')
    if i == 800: 
        print('800')
    predictions_classes = resample(test_df_filtered['predictions_classes'], replace=True, n_samples=sample_size, random_state=i)
    predictions = resample(test_df_filtered['predictions'], replace=True, n_samples=sample_size, random_state=i)
    true_y = resample(test_df_filtered['Genetic sex_0.0_integers'], replace=True, n_samples=sample_size, random_state=i)
    auc_score_samples.append(roc_auc_score(true_y, predictions))
    precision, recall, _ = precision_recall_curve(true_y, predictions)
    prc_score_samples.append(auc(recall, precision))
    
print('AUC All Images', np.percentile(auc_score_samples, [2.5, 97.5]))
print('PRC All Images', np.percentile(prc_score_samples, [2.5, 97.5]))


print('Non-parametric bootstrapping - Left Images')

auc_score_samples = []
prc_score_samples = []

for i in range(0, 2000): 
    if i == 100: 
        print('100')
    if i == 500: 
        print('500')
    if i == 800: 
        print('800')
    predictions_classes = resample(df_left['predictions_classes'], replace=True, n_samples=sample_size, random_state=i)
    predictions = resample(df_left['predictions'], replace=True, n_samples=sample_size, random_state=i)
    true_y = resample(df_left['Genetic sex_0.0_integers'], replace=True, n_samples=sample_size, random_state=i)
    auc_score_samples.append(roc_auc_score(true_y, predictions))
    precision, recall, _ = precision_recall_curve(true_y, predictions)
    prc_score_samples.append(auc(recall, precision))
    
print('AUC Left Images', np.percentile(auc_score_samples, [2.5, 97.5]))
print('PRC Left Images', np.percentile(prc_score_samples, [2.5, 97.5]))


print('Non-parametric bootstrapping - Right Images')

auc_score_samples = []
prc_score_samples = []

for i in range(0, 2000): 
    if i == 100: 
        print('100')
    if i == 500: 
        print('500')
    if i == 800: 
        print('800')
    predictions_classes = resample(df_right['predictions_classes'], replace=True, n_samples=sample_size, random_state=i)
    predictions = resample(df_right['predictions'], replace=True, n_samples=sample_size, random_state=i)
    true_y = resample(df_right['Genetic sex_0.0_integers'], replace=True, n_samples=sample_size, random_state=i)
    auc_score_samples.append(roc_auc_score(true_y, predictions))
    precision, recall, _ = precision_recall_curve(true_y, predictions)
    prc_score_samples.append(auc(recall, precision))

    
print('AUC Right Images', np.percentile(auc_score_samples, [2.5, 97.5]))
print('PRC Right Images', np.percentile(prc_score_samples, [2.5, 97.5]))



auc_score_samples = []
prc_score_samples = []

for i in range(0, 2000): 
    if i == 100: 
        print('100')
    if i == 500: 
        print('500')
    if i == 800: 
        print('800')
    predictions_classes = resample(mean_df['predictions_classes'], replace=True, n_samples=sample_size, random_state=i)
    predictions = resample(mean_df['predictions'], replace=True, n_samples=sample_size, random_state=i)
    true_y = resample(mean_df['Genetic sex_0.0_integers'], replace=True, n_samples=sample_size, random_state=i)
    auc_score_samples.append(roc_auc_score(true_y, predictions))
    precision, recall, _ = precision_recall_curve(true_y, predictions)
    prc_score_samples.append(auc(recall, precision))
    
print('AUC Mean Images', np.percentile(auc_score_samples, [2.5, 97.5]))
print('PRC Right Images', np.percentile(prc_score_samples, [2.5, 97.5]))








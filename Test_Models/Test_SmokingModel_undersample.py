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
test_df_filtered = test_df[test_df['Smoking'].notna()]

test_df_filtered['Smoking_binarized'] = test_df_filtered['Smoking'].map({'current': 'smoking', 'ideal': 'non_smoking', 'intermediate' : 'non_smoking'})
test_df_filtered['Smoking_binarized'] = test_df_filtered['Smoking_binarized'].map({'smoking': 1, 'non_smoking': 0})


#Load Model 
filepath = '/home/andreap/fundusdata_2/results_retina_alllayers/smoking_undersampling.hdf5'
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


#All Images Predictions

print('AUC All Images:', roc_auc_score(test_df_filtered['Smoking_binarized'], test_df_filtered['predictions']))
precision, recall, _ = precision_recall_curve(test_df_filtered['Smoking_binarized'], test_df_filtered['predictions'])
auc_score = auc(recall, precision)
print('AUROC All Images:', auc_score)


#Left Eye Images Predictions
df_left = test_df_filtered[test_df_filtered['left_right'] == '21015']
print('AUC Left:', roc_auc_score(df_left['Smoking_binarized'], df_left['predictions']))
precision, recall, _ = precision_recall_curve(df_left['Smoking_binarized'], df_left['predictions'])
auc_score = auc(recall, precision)
print('AUROC Left Images:', auc_score)


#Right Eye Images Predictions
df_right = test_df_filtered[test_df_filtered['left_right'] == '21016']
precision, recall, _ = precision_recall_curve(df_right['Smoking_binarized'], df_right['predictions'])
auc_score = auc(recall, precision)
print('AUROC Right Images:', auc_score)

print('AUC Right:', roc_auc_score(df_right['Smoking_binarized'], df_right['predictions']))



mean_df = test_df_filtered.groupby('eid').mean()
mean_df['predictions_classes'] = np.where(mean_df.predictions > 0.5, 1, 0)

print('AUC Mean Images:', roc_auc_score(mean_df['Smoking_binarized'], mean_df['predictions']))
precision, recall, _ = precision_recall_curve(mean_df['Smoking_binarized'], mean_df['predictions'])
auc_score = auc(recall, precision)
print('AUROC Mean Images:', auc_score)

eid_ethnic = test_df_filtered[['eid', 'Ethnic background_0.0']].drop_duplicates()
mean_df = pd.merge(mean_df, eid_ethnic, left_on = 'individual_id', right_on = 'eid')

british_irish = mean_df[(mean_df['Ethnic background_0.0'] == 'British')|(mean_df['Ethnic background_0.0'] == 'Irish')]
eid = british_irish['eid']
other_than_british = mean_df[~mean_df.eid.isin(eid)]

print('AUC British/Irish:', roc_auc_score(british_irish['Smoking_binarized'], british_irish['predictions']))


print('AUC Other than British/Irish:', roc_auc_score(other_than_british['Smoking_binarized'], other_than_british['predictions']))



eid_gender = test_df_filtered[['eid', 'Genetic sex_0.0']].drop_duplicates()
mean_df = pd.merge(mean_df, eid_gender, left_on = 'individual_id', right_on = 'eid')
male_df = mean_df[mean_df['Genetic sex_0.0'] == 'Male']
female_df = mean_df[mean_df['Genetic sex_0.0'] == 'Female']



print('AUC Male:', roc_auc_score(male_df['Smoking_binarized'], male_df['predictions']))

print('AUC Female:', roc_auc_score(female_df['Smoking_binarized'], female_df['predictions']))


age_between39_50 = mean_df[(mean_df['Age'] > 39) & (mean_df['Age'] <= 50)]
age_above_50 = mean_df[mean_df['Age'] > 50]


print('AUC Mean Images Between 39 and 50:', roc_auc_score(age_between39_50['Smoking_binarized'], age_between39_50['predictions']))



print('AUC Mean Images Above 50:', roc_auc_score(age_above_50['Smoking_binarized'], age_above_50['predictions']))


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
    true_y = resample(test_df_filtered['Smoking_binarized'], replace=True, n_samples=sample_size, random_state=i)
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
    true_y = resample(df_left['Smoking_binarized'], replace=True, n_samples=sample_size, random_state=i)
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
    true_y = resample(df_right['Smoking_binarized'], replace=True, n_samples=sample_size, random_state=i)
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
    true_y = resample(mean_df['Smoking_binarized'], replace=True, n_samples=sample_size, random_state=i)
    auc_score_samples.append(roc_auc_score(true_y, predictions))
    precision, recall, _ = precision_recall_curve(true_y, predictions)
    prc_score_samples.append(auc(recall, precision))
    
print('AUC Mean Images', np.percentile(auc_score_samples, [2.5, 97.5]))
print('PRC Right Images', np.percentile(prc_score_samples, [2.5, 97.5]))
# MasterThesis
<br />
In this project I predicted cardiovascular risk factors from retinal fundus images from the UK Biobank data set.
<br />
In the folder ClassificationTaskModel, the prediction models for the categorical variables Gender and Smoking Status can be found. <br />
The Gender prediction model is a binary classification model (Female, Male).
<br />
For the smoking status (binary classification model), different models (oversampling, undersampling, weighting inversely proportional to frequency) have been tested to counteract class imbalance.
<br />
In the folder RegressionTaskModel, the prediction model for continuous variables (age, sbp, dbp, hbA1c, cholesterol, bmi) can be found. 
<br />
In the folder Test_Models, the performance of the regression model and classification models are evaluated. 
The regression models are evaluated by calculating the MAE and R2 score. 
The classification models are evaluated by calculating the AUC score.
To evalulate the uncertainty of the estimate, non-parametric bootstrapping is applied. 
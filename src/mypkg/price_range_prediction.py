#!/usr/bin/env python
# coding: utf-8

# Final Project
# Categorization of Houses into Different Price Range using ML Algorithms from American Housing Survey 2017 Dataset

# The main goal of this project is to predict the range of selling price of house with a high degree of predictive accuracy using various 
# Machine Learning methods. Given house sale data or explanatory variable such as number of bedrooms, number of bathrooms in unit, housing 
# cost, annual commuting cost etc, we build our model. Next, the model is evaluated with respect to test data, and plot the prediction and 
# coefficients.

# For my project, I have prepared two types of the same file - one .py and other .ipynb. The .py version is for testing using pytest. I am 
# applying different machine learning algorithms and using a big dataset. Therefore, my .ipynb file became too large (around 90MB) which 
# cannot be uploaded in github repo as it is. Therefore, I prepared a PDF copy of .ipynb file with all outputs that got generated, so that 
# outputs of program are visible. Also, I cleared all outputs for .ipynb file and uploaded that as well. All the relevant documents along 
# with the .ipynb with all generated outputs is present in google drive -
# https://drive.google.com/drive/u/0/folders/1Or1xQ5GVPU1sCB3hY7V5pAKYYp-aP2Nd

# Importing necessary packages
import pandas as pd
import numpy as np
from numpy import argmax
import re
import copy
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
#pd.set_option('display.max_rows', 1000)
#pd.set_option('display.max_columns', 1000)
import math
from subprocess import call
from IPython.display import Image
from IPython.display import display
import warnings; warnings.simplefilter('ignore')

# Learning Libraries
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
#from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression  


# Loading the dataset
# We are using American Housing Survey 2017 data https://www.census.gov/programs-surveys/ahs/data/2017/ahs-2017-public-use-file--puf-/ahs-
# 2017-national-public-use-file--puf-.html (household.csv in AHS 2017 National PUF v3.0 CSV.zip). Since the dataset is very big, I am just 
# providing the link. It could not be uploaded in github repo. There is another csv file called AHSDICT_15NOV19_21_17_31_97_S.csv that 
# consist of the mapping information of each feature name to their actual meaning and data type information. This file is already present 
# in github repo. In the AHS microdata, the basic unit is an individual housing unit. Each record shows most of the information associated 
# with a specific housing unit or individual, except for data items that could be used to personally identify that housing unit or 
# individual. Our dataset comprises of housing data features like TOTROOMS(Number of rooms in unit), PERPOVLVL(Household income as percent 
# of poverty threshold (rounded)), COMCOST(Total annual commuting cost), JBATHROOMS(Number of bathrooms in unit), UNITSF(Square footage of 
# unit), JGARAGE(Flag indicating unit has a garage or carport), JFIREPLACE(Flag indicating unit has a useable fireplace) etc., and target 
# column as MARKETVAL(Current market value of unit) to evaluate model and also check which amongst all features is the most correlated 
# feature for price predication.
data = pd.read_csv("household.csv")
headings = pd.read_csv("AHSDICT_15NOV19_21_17_31_97_S.csv", encoding = "ISO-8859-1")


# Data Cleaning
# Formatting the columns to check
col_to_check = data.columns
data[col_to_check] = data[col_to_check].replace({'\'':''}, regex=True)


# The column CONTROL is not relevant to our problem, so we can remove that 
col_to_remo = ['CONTROL']
data = data.drop(col_to_remo, axis = 1)


# Replace all Not Applicable/No Response values with Nan for further processing
L = ['-6', -6, '-9', -9, 'M', 'N']
data = data.replace(L, np.nan)


# Getting rid of non relevant values
for c in list(data.columns): 
    nan = (len(data) - data[c].count())/(len(data))
    if nan >= 0.85:
        del data[c]


# Target column
data['MARKETVAL'].describe()
data = data[pd.notnull(data['MARKETVAL'])]


# Final dimension of dataset to be used
indexNames = data[ data['MARKETVAL'] < 50000 ].index
data.drop(indexNames , inplace=True)


# Checking distribution of data
plt.hist(data['MARKETVAL'], bins = int(180/5), color = 'blue', edgecolor = 'black')


# Dividing the dataset into numerical and categorical features
col_o = list(data.columns)
numeric = []
categorical = []
e = []
for c in col_o:
    j = 0
    if c[0]=='J':
        j = 1
        c = c[1:]
    h = headings.loc[headings['Variable']== c]['TYPE'].tolist()
    if h != []:
        if (h[0] == 'Character'):
            if j == 0:
                categorical.append(c)
            elif j == 1:
                categorical.append('J' + c)
        elif (h[0] == 'Numeric'):
            if j == 0:
                numeric.append(c)
            elif j == 1:
                numeric.append('J' + c)
        else:
            if j == 0:
                e.append(c)
            elif j == 1:
                e.append('J' + c)


# Defining data_numeric which only has numerical features
data_numeric = data.drop(categorical, axis = 1)


# Getting rid of all NaN entries in data_numeric
data_numeric = data_numeric.fillna(data_numeric.mean())
for i in numeric:
    if math.isnan(float(data_numeric[i].mean())):
        # drop the unnecessary columns
        print("Dropping the Column: ", i)
        data_numeric = data_numeric.drop(i, axis = 1)
    else:
        data_numeric[i] = data_numeric[i].fillna(data_numeric[i].mean())


# Defining data_categorical which only has categorical features
data_categorical = data.drop(numeric, axis = 1)


# Getting rid of all NaN entries in data_catagorical
for i in categorical:
    # dict to store counts of each unique value occurring for each feature
    freq = {}
    for j in data_categorical[i]:
        if (j in freq): 
            freq[j] += 1
        else: 
            freq[j] = 1
    freq_sorted = sorted(freq, key=freq.get, reverse=True)
    
    # if the most frequent value is Nan
    if math.isnan(float(freq_sorted[0])):
        # if Nan is not the only value for that feature, then use the next most frequent value to replace Nan
        if len(freq_sorted) > 1:
            mode_val = freq_sorted[1]
            data_categorical[i] = data_categorical[i].fillna(mode_val)
        # if Nan is the only value for that feature, then drop the column
        else: 
            # drop the unnecessary columns
            print("Dropping the Column: ", i)
            data_categorical = data_categorical.drop(i, axis = 1)
    else:
        mode_val = freq_sorted[0]
        data_categorical[i] = data_categorical[i].fillna(mode_val)
        
#print(data_categorical.isnull().sum())


# Concatenate numerical and categorical data
clean_data = pd.concat([data_numeric, data_categorical], axis=1, sort=False)


# Remove duplicate columns after concatenation
clean_data = clean_data.iloc[:,~clean_data.columns.duplicated()]


# Finally check if finally cleaned data is free of null values
#print(clean_data.isnull().sum())


# Correlation matrix to check which amongst all features is the most correlated feature for price prediction
corr_matrix=clean_data.corr()
corr_matrix["MARKETVAL"].sort_values(ascending=False)
#corr_matrix.style.background_gradient(cmap='coolwarm').set_precision(2)


# The dataset was cleaned to make it free from erroneous or irrelevant data. By filling up missing values, removing rows and reducing data 
# size, the final dataset was (36358 rows X 1007 columns).


# Data Split
# Now we will separate the Test Data and Train Data. Will keep 30% of the data for Testing purpose and rest for training purpose.
# Separating out the target
y = clean_data['MARKETVAL']
# Separating out the features
x = clean_data.drop('MARKETVAL', axis = 1)
# Splitting dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# Separating the numeric and categorical features
col_o = list(x.columns)
numeric = []
categorical = []
e = []
for c in col_o:
    j = 0
    if c[0]=='J':
        j = 1
        c = c[1:]
    h = headings.loc[headings['Variable']== c]['TYPE'].tolist()
    if h != []:
        if (h[0] == 'Character'):
            if j == 0:
                categorical.append(c)
            elif j == 1:
                categorical.append('J' + c)
        elif (h[0] == 'Numeric'):
            if j == 0:
                numeric.append(c)
            elif j == 1:
                numeric.append('J' + c)
        else:
            if j == 0:
                e.append(c)
            elif j == 1:
                e.append('J' + c)


# Training and Test Data for numeric features
X_train_numeric = X_train.drop(categorical, axis = 1)
X_test_numeric = X_test.drop(categorical, axis = 1)


# Scalar Transform the Numeric Variable
scaler = MinMaxScaler()#StandardScaler()
scaler.fit(X_train_numeric)


# Apply transform to both the training set and the test set. Nor_numerical = pd.DataFrame(scaler.transform(numerical))
X_train_numeric_trans = pd.DataFrame(scaler.transform(X_train_numeric))
X_test_numeric_trans = pd.DataFrame(scaler.transform(X_test_numeric))

X_train_numeric_trans.columns = X_train_numeric.columns
X_test_numeric_trans.columns = X_test_numeric.columns


# Training and Test Data for catagorical features
X_train_categorical = X_train.drop(numeric, axis = 1)
X_test_categorical = X_test.drop(numeric, axis = 1)


# Resetting the index to avoid NaN on concatenation
X_train_numeric_trans.reset_index(drop=True, inplace=True)
X_train_categorical.reset_index(drop=True, inplace=True)

X_test_numeric_trans.reset_index(drop=True, inplace=True)
X_test_categorical.reset_index(drop=True, inplace=True)


# Concatenating numeric and catagorical features
X_train = pd.concat([X_train_numeric_trans, X_train_categorical], axis=1, sort=False)
X_test = pd.concat([X_test_numeric_trans, X_test_categorical], axis=1, sort=False)


# Remove duplicate columns after concatenation
X_train = X_train.iloc[:,~X_train.columns.duplicated()]
X_test = X_test.iloc[:,~X_test.columns.duplicated()]


# Final formatting before applying algorithms
y_train = pd.DataFrame(y_train) 
y_train = y_train.reset_index(drop=True)
y_train_int = y_train.astype(int)

y_test = pd.DataFrame(y_test)
y_test = y_test.reset_index(drop=True)
y_test_int = y_test.astype(int)

# Since, we want to classify houses into different price ranges, we will need to perform feature encoding for various price ranges.
# Function to assign code for different price ranges
def price_range(price):
    if price < 100000:
        return 1
    elif price >= 100000 and price < 250000:
        return 2
    elif price >= 250000 and price < 500000:
        return 3
    elif price >= 500000 and price < 750000:
        return 4
    elif price >= 750000 and price < 1000000:
        return 5
    elif price >= 1000000 and price < 1250000:
        return 6
    elif price >= 1250000:
        return 7
    else:
        print(price)
        return 13


# Get the price range encoded field in y dataset
y_train['MARKETVAL'] = y_train_int['MARKETVAL'].apply(price_range)
y_test['MARKETVAL'] = y_test_int['MARKETVAL'].apply(price_range)
    
    
# From below histograms it can be seen that most houses fall in the MARKETVAL range of 100000 to 250000
# Distribution of data in y training dataset
plt.hist(y_train['MARKETVAL'], bins = int(180/5), color = 'blue', edgecolor = 'black')


# Distribution of data in y testing dataset
plt.hist(y_test['MARKETVAL'], bins = int(180/5), color = 'blue', edgecolor = 'black')


# One Hot Encoding to represent categorical variables as binary vectors
le = LabelEncoder()
le.fit(y_train)
y_train_le = le.transform(y_train['MARKETVAL'])#.reshape(-1, 1)
#y_test_le = le.transform(y_train['MARKETVAL'])#.reshape(-1, 1)

oh = OneHotEncoder(sparse=False)
y_train_le = y_train_le.reshape(len(y_train_le), 1)
oh.fit(y_train_le)

y_train_oh = oh.transform(y_train_le)

# Performance Measures
# The function accuracy is used to calculate accuracy scores for both training and testing dataset for different ML models. The function
# train_model is used to train and fit the data for different classifier models.
# Calculating accuracy score for training and testing datasets
def accuracy(X_train, X_test, y_train, y_test, model): 
    y_pred_train = model.predict(X_train)
    train_accuracy = accuracy_score(y_train.values, y_pred_train)
    print(f"Train accuracy: {train_accuracy:0.2%}")
    
    y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test.values, y_pred_test)
    print(f"Test accuracy: {test_accuracy:0.2%}")

    
# Function to train data based on different classifiers
def train_model(X_train, X_test, classifier, **kwargs):
    model = classifier(**kwargs)
    model.fit(X_train, y_train)
    return model


# Algorithms Implemented
# In this project, my aim was to implement algorithms which will be able to learn and classify the new observations to correct house price 
# ranges. I decided to use below machine learning algorithms for the same-
# • Random Forest (RandomForestClassifier)
# • Logistic Regression (LogisticRegression)
# • K-Nearest Neighbor (KNeighborsClassifier)
# • Decision Tree (DecisionTreeClassifier)


# Test accuracy for all models for comparison later
accuracy_val = []
# List of Algorithms Mames
classifiers = ['Random Forest', 'Logistic Regression', 'Knn (7 Neighbors)', 'Decision Tree']


# Using Random Forest Classifier
# The random forest is a model made up of many decision trees. Rather than just simply averaging the prediction of trees, this model uses 
# two key concepts that gives it the name random:
# • Random sampling of training data points when building trees
# • Random subsets of features considered when splitting nodes
# The random forest combines hundreds or thousands of decision trees, trains each one on a slightly different set of the observations, 
# splitting nodes in each tree considering a limited number of the features. The final predictions of the random forest are made by 
# averaging the predictions of each individual tree.
model = train_model(X_train, y_train_oh, RandomForestClassifier, n_estimators=200, random_state=20)
test_accuracy_val = accuracy(X_train, X_test, y_train, y_test, model)
accuracy_val.append(test_accuracy_val)

# Results: With RandomForestClassifier, the accuracy score were as below:
# Training Accuracy – 100.00%
# Testing Accuracy – 55.86%


# I also plotted a bar graph representing the top 10 features based on their importance in determining the house price range.
pd.Series(model.feature_importances_, x.columns).sort_values(ascending=True).nlargest(10).plot.barh(align='center')


# Using Logistic Regression
# Logistic regression is one of the most fundamental and widely used Machine Learning Algorithms. Logistic regression is not a regression 
# algorithm but a probabilistic classification model. Multi class classification is implemented by training multiple logistic regression 
# classifiers, one for each of the K classes in the training dataset.
model = train_model(X_train, y_train, LogisticRegression,solver='lbfgs')
test_accuracy_val = accuracy(X_train, X_test, y_train, y_test, model)
accuracy_val.append(test_accuracy_val)

# Results: With LogisticRegression, the accuracy score were as below:
# Training Accuracy – 47.08%
# Testing Accuracy – 46.68%


# Using kNN Classifier
# KNN or k-nearest neighbours is the simplest classification algorithm. This classification algorithm does not depend on the structure of 
# the data. Whenever a new example is encountered, its k nearest neighbours from the training data are examined. Distance between two 
# examples can be the euclidean distance between their feature vectors. The majority class among the k nearest neighbours is taken to be 
# the class for the encountered example.
model = train_model(X_train, y_train, KNeighborsClassifier, n_neighbors=7)
test_accuracy_val = accuracy(X_train, X_test, y_train, y_test, model)
accuracy_val.append(test_accuracy_val)

# Results: With KNeighborsClassifier, the accuracy score were as below:
# Training Accuracy – 60.04%
# Testing Accuracy – 46.89%

# Using Decision Tree Classifier
# Decision tree classifier is a systematic approach for multiclass classification. It poses a set of questions to the dataset (related to 
# its attributes/features). The decision tree classification algorithm can be visualized on a binary tree. On the root and each of the 
# internal nodes, a question is posed and the data on that node is further split into separate records that have different characteristics. 
# The leaves of the tree refer to the classes in which the dataset is split.
model = train_model(X_train, y_train, DecisionTreeClassifier, max_depth=8)
test_accuracy_val = accuracy(X_train, X_test, y_train, y_test, model)
accuracy_val.append(test_accuracy_val)

# Results: With DecisionTreeClassifier, the accuracy score were as below:
# Training Accuracy – 65.45%
# Testing Accuracy – 59.83%


# Conclusion
# The purpose of this project was correlate and compare the above mentioned ML algorithms in order to check their performances.

# Create a dataframe from accuracy results
summary = pd.DataFrame({'Test Accuracy':accuracy_val}, index=classifiers)       
summary

# For this particular problem, the algorithm with best accuracy value is DecisionTreeClassifier with test accuracy score of 59.83% and 
# therefore it can be considered as a good classifier algorithm for house price range prediction problem. Also, the RandomForestClassifier 
# is close enough with 55.86% accuracy score. I have tried tuning each algorithm with different hyper-parameter values and finally kept the 
# best results for each. In this project we can say that in machine learning problems data processing and tuning makes the model more 
# accurate and efficient compare to non processed data. It also makes simple models quite accurate.

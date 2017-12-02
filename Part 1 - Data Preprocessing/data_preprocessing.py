# Data Processing

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pandas

# Importing dataset
dataset = pandas.read_csv('Data.csv')
print(dataset)
# data selection: [: ,:] 
# this is defining columns then rows
# colons (:) denote taking all 
# :-1 means to take all and remove from end (so this is all but the last entry)
# specifying integer is to target that entry
# 1:3 marks to take a range. This is inclusive of hte first and exclusive of the last
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])


# encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_x = LabelEncoder()
x[:, 0] = labelEncoder_x.fit_transform(x[:, 0])
hotEncoder_x = OneHotEncoder(categorical_features = [0])
x = hotEncoder_x.fit_transform(x).toarray()


labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)


# Splitting data into Training and Test sets
# The training set is used to teach the machine how to understand 
# the data to make predictions on new data sources
# So if the machine is learning correctly then it will be able to 
# accurately predict the values in the test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
scale_x = StandardScaler()
scale_x = scale_x.fit(x_train)

x_train = scale_x.transform(x_train)
x_test = scale_x.transform(x_test)

# OR if to not scale the categorical data (countries)
# we can set the fit data to a range (as with the categorical data)
# and only transform that set so we can maintain the original reference
# scaling all dfata is a go d solution for guaranteeing standardised data everywhere`
# scale_x = scale_x.fit(x_train[:, 3:5])
# x_train[:, 3:5] = scale_x.transform(x_train[:, 3:5])



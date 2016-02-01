# Data Imports
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

# Math
import math

# Plot imports

#import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set_style('whitegrid')

# Machine Learning Imports
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

# For evaluating our ML results
from sklearn import metrics

# season DataFrame is simply the regular/real seaon connected to the post/play off data
real_season = pd.read_csv('passes.csv')
season = pd.read_csv('combined.csv')
post_season = pd.read_csv('passes_post.csv')


# The average is found to convert from integer to decimals since logistic regression requires continuity.
def find_avg(x):
    return x/16.0


# Dummy variables to remove collinearity, e.g. If a variable can be one of 3 (WR, RB, TE), if it is neither RB or TE, 
# it can only be WR so we can aurgue they are correlated
receiver_pos_dummies = pd.get_dummies(season['Receiver Pos'])
down_dummies = pd.get_dummies(season['Down'])
down_dummies.columns = ['Down 1', 'Down 2', 'Down 3', 'Down 4']
team_dummies = pd.get_dummies(season['Team'])
receiver_id_dummies = pd.get_dummies(season['Receiver ID'])
qb_dummies = pd.get_dummies(season['QB'])

# Prepare the dataframe
training_data = season.drop(['Receiver Pos', 'Down', 'Team', 'Intercepted', 'Success','Receiver ID', 'QB'], axis=1)
dummies = pd.concat([qb_dummies, receiver_pos_dummies, receiver_id_dummies, down_dummies, team_dummies], axis=1)
training_data = pd.concat([training_data, dummies], axis=1)

# Apply avg and drop columns
training_data['QB Sacks'] = training_data['QB Sacks'].apply(find_avg)
training_data['QB Fumbles'] = training_data['QB Fumbles'].apply(find_avg)
training_data['QB Rush Att'] = training_data['QB Rush Att'].apply(find_avg)

# Remove columns to get rid of collinearity
training_data = training_data.drop(['B.Roethlisberger','RB', 1, 'Down 4', 'PIT'], axis=1)


# Season succesful passes
Y_season = season['Success']

'''Reg season length 0 - 17821'''

Y = np.ravel(Y_season)

log_model = LogisticRegression()
log_model.fit(training_data, Y)

# Model accuracy based on all of the data
print 'Model 1'
print log_model.score(training_data, Y)

#Null Error Rate
print 1 - Y.mean()
print '\n'

# Coeff and the impactfulness on the model
coeff_df = DataFrame(zip(training_data.columns, np.transpose(log_model.coef_)))
coeff_df.to_csv('result1.csv')


#Split the dataset inot regular and post season data
X_train = training_data[:17822]
X_test = training_data[17822:]

y_train = np.ravel(Y_season[:17822])
y_test = np.ravel(Y_season[17822:])


log_model2 = LogisticRegression()
log_model2.fit(X_train, y_train)

class_predict = log_model2.predict(X_test)

# Accuracy and null error rate
print 'Model 2'
print metrics.accuracy_score(y_test, class_predict)
print 1 - y_test.mean()
print '\n'

# Coeff and the impactfulness on the model
coeff_df = DataFrame(zip(X_train.columns, np.transpose(log_model2.coef_)))
coeff_df.to_csv('result2.csv')


log_model3 = LogisticRegression(penalty='l1')
log_model3.fit(X_train, y_train)

class_predict = log_model3.predict(X_test)

# Accuracy and null error rate
print 'Model 3'
print metrics.accuracy_score(y_test, class_predict)
print 1 - y_test.mean()
print '\n'

# Coeff and the impactfulness on the model
coeff_df = DataFrame(zip(X_train.columns, np.transpose(log_model3.coef_)))
coeff_df.to_csv('result3.csv')

log_model4 = LogisticRegression(penalty='l2')
log_model4.fit(X_train, y_train)

class_predict = log_model4.predict(X_test)

# Accuracy and null error rate
print 'Model 4'
print metrics.accuracy_score(y_test, class_predict)
print 1 - y_test.mean()
print '\n'

# Coeff and the impactfulness on the model
coeff_df = DataFrame(zip(X_train.columns, np.transpose(log_model4.coef_)))
coeff_df.to_csv('result4.csv')
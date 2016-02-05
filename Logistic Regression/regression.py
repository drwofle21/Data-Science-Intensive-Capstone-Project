# Data Imports
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

# Math
import math

# Plot imports

import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set_style('whitegrid')

# Machine Learning Imports
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

# For evaluating our ML results
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve

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
print 'Model Logistic'
print log_model.score(training_data, Y)

#Null Error Rate
print 1 - Y.mean()
print '\n'

# Coeff and the impactfulness on the model
coeff_df = DataFrame(zip(training_data.columns, np.transpose(log_model.coef_)))
coeff_df.to_csv('result1.csv')

'''Model Naive Bayes'''
gnb = GaussianNB()
gnb.fit(training_data, Y)

# Model accuracy based on all of the data
print 'Model Naive Bayes'
print gnb.score(training_data, Y)

#Null Error Rate
print 1 - Y.mean()
print '\n'

# Coeff and the impactfulness on the model
#coeff_df = DataFrame(zip(training_data.columns, np.transpose(gnb.coef_)))
#coeff_df.to_csv('resultgnb.csv')

'''Model Decision Tree'''
clf = tree.DecisionTreeClassifier()
clf.fit(training_data, Y)

# Model accuracy based on all of the data
print 'Model Decision Tree'
print clf.score(training_data, Y)

#Null Error Rate
print 1 - Y.mean()
print '\n'

# Coeff and the impactfulness on the model
#coeff_df = DataFrame(zip(training_data.columns, np.transpose(clf.coef_)))
#coeff_df.to_csv('resultclf.csv')

#===================================================================================

#Split the dataset inot regular and post season data
X_train = training_data[:17822]
X_test = training_data[17822:]

y_train = np.ravel(Y_season[:17822])
y_test = np.ravel(Y_season[17822:])

#=======================================================================================


log_model2 = LogisticRegression()
log_model2.fit(X_train, y_train)

class_predict = log_model2.predict(X_test)

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)


df_confusion = pd.crosstab(y_test, class_predict, rownames=['Actual'], colnames=['Predicted'], margins=True)

df_conf_norm = df_confusion / df_confusion.sum(axis=1)

# Accuracy and null error rate
print 'Model 2'
print metrics.accuracy_score(y_test, class_predict)
print 1 - y_test.mean()
print '\n'

# Coeff and the impactfulness on the model
coeff_df = DataFrame(zip(X_train.columns, np.transpose(log_model2.coef_)))
coeff_df.to_csv('result2.csv')

# Compute confusion matrix
cm = confusion_matrix(y_test, class_predict)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)

plot_confusion_matrix(df_confusion)

fpr, tpr, thresholds = roc_curve(y_test, class_predict, pos_label=2)

#======================================================================================
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



plt.show()
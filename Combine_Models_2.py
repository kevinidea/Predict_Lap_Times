import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from Combine_Models import combine_models
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split, KFold
from sklearn.ensemble import BaggingRegressor

#Load all models built previously
model1 = joblib.load('Linear_Regression_Model.pkl')
model2 = joblib.load('SVR_Model.pkl')
model3 = joblib.load('Kernel_Ridge_Model.pkl')
model4 = joblib.load('Lasso_Model.pkl')
model5 = joblib.load('Linear_Ridge_Model.pkl')
model6 = joblib.load('Bayesian_Ridge_Model.pkl')
model7 = joblib.load('SGD_Model.pkl')
model8 = joblib.load('Random_Forest_Model.pkl')

#Read the data for training final super model
data = pd.read_csv("cars_data.csv")

#convert all lap times into seconds
pattern = data['Lap Time'].str.extract(r"(\d+)\:(\d+\.\d+)")
condition = (data['Lap Time'].str.contains(r"(\d+)\:(\d+\.\d+)")) & (data['Lap Time'].notnull())
#Copy Lap Time column to Lap_Time
data['Lap_Time'] = data['Lap Time']
#Override the lap_time that match the pattern with transformed lap time
data.loc[condition, 'Lap_Time'] = pattern.loc[condition, 0].astype(float)*60 + pattern.loc[condition, 1].astype(float)
#Delete the Original lap time column
data = data.drop('Lap Time', axis =1)

#Encode categorical variable(s) into boolean dummy variable(s)
def transform_categorical_variables(data, cols, drop_categorical_columns=False):
    vec = DictVectorizer()
    mkdict = lambda row: dict((col, row[col]) for col in cols)
    vecData = pd.DataFrame(vec.fit_transform(data[cols].apply(mkdict, axis=1)).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if drop_categorical_columns is True:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    else:
        data = data.join(vecData)
    return data

#Dummy code categorical variable(s)
data2 = transform_categorical_variables \
(data, ['Track', 'Condition', 'Car Brand', 'Layout'], drop_categorical_columns = True)

#Choose features to run model
x = data2.drop(['Car', 'Ranking', 'Lap_Time', 'Country', 'Car Type', 'Engine',\
'Diesel', 'Gear', 'HP Per Liter', 'HP Per Ton' ], axis =1)
y = data2['Lap_Time']

#Separate training and testing data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2, random_state =6)

#Build the final model based on predictions from all 8 previous models
combine_predictions = combine_models(model1, model2, model3, model4, model5, model6, model7, model8)
model = BaggingRegressor()
super_model = Pipeline([('combine_models', combine_predictions), ('model', model)])
#super_model.fit(x_train, y_train)

#Evaluate its performance
print "CROSS VALIDATION ERROR"
kf = KFold(len(x_train), n_folds = 10)
cv_errors = []
for train_idx, test_idx in kf:
    x_cross_train = x_train.iloc[train_idx, :]
    y_cross_train = y_train.iloc[train_idx]
    super_model.fit(x_cross_train, y_cross_train)
    x_cross_test = x_train.iloc[test_idx, :] #x is a dataframe
    y_cross_test = y_train.iloc[test_idx] #y is a series
    y_predicted = super_model.predict(x_cross_test)
    mae = mean_absolute_error(y_predicted, y_cross_test)
    cv_errors.append(mae)
    print mae
print "--------------------------------"
print "CROSS VALIDATION MEAN ABSOLUTE ERROR with 10 folds for:"
print "Super Model: mean = %.4f, std = %.4f" %(np.mean(cv_errors), np.std(cv_errors))
print "--------------------------------"
print "TESTING ERROR"
y_predicted = super_model.predict(x_test)
print 'Mean absolute error: %.4f' %(mean_absolute_error(y_predicted,y_test))
print 'Mean squared error: %.4f' %(mean_squared_error(y_predicted, y_test))
print 'R2 error: %.4f' %(r2_score(y_predicted, y_test))
print "--------------------------------"
print "TRAINING ERROR"
y_predicted = super_model.predict(x_train)
print 'Mean absolute error: %.4f' %(mean_absolute_error(y_predicted,y_train))
print 'Mean squared error: %.4f' %(mean_squared_error(y_predicted, y_train))
print 'R2 error: %.4f' %(r2_score(y_predicted, y_train))

#Fit the super model with the entire data set
super_model.fit(x,y)
print "--------------------------------"
print "FINAL TRAINING ERROR"
y_predicted = super_model.predict(x)
print 'Mean absolute error: %.4f' %(mean_absolute_error(y_predicted,y))
print 'Mean squared error: %.4f' %(mean_squared_error(y_predicted, y))
print 'R2 error: %.4f' %(r2_score(y_predicted, y))

# save to pickle
joblib.dump(super_model, 'Super_Model.pkl', compress=1)

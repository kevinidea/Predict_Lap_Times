import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.grid_search import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
import time

start_time = time.time()

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
#All categorical variable(s): 'Condition', 'Track', 'Country', 'Layout', 'Car Type', 'Engine', 'Car Brand'
data2 = transform_categorical_variables \
(data, ['Track', 'Condition', 'Car Brand', 'Layout'], drop_categorical_columns = True)

#Choose features to run model
#All numerical variable(s): 'Year Model', 'HP', 'Torque', 'Weight', 'Turbocharged',
# 'Diesel', 'Gear', 'Displacement', 'HP Per Liter', 'HP Per Ton', 'Top Speed'
x = data2.drop(['Car', 'Ranking', 'Lap_Time', 'Country', 'Car Type', 'Engine',\
'Diesel', 'Gear', 'HP Per Liter', 'HP Per Ton' ], axis =1)
y = data2['Lap_Time']
#print x.columns

#Separate training and testing data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.1, random_state =101)

#Normalize and scale x before tuning many models
scaler = MinMaxScaler(feature_range=(-1,1))

#####Develop models with 10 Kfolds cross validation and 100 randomized grid searches
#Set up cross validation parameters for many models
scoring = 'mean_absolute_error'
cv = 10
n_iter = 100
distribution = [10**i for i in range (-7,3)]

######Develop Basic Linear Model
lr = LinearRegression()
lr.fit(x_train, y_train)
#Save the model
joblib.dump(lr, 'Linear_Regression_Model.pkl', compress=1)


#####Develop SVR Model
svr = Pipeline([
                ('scaler', scaler),
                ('SVR',SVR(C=10, gamma =1, kernel = 'linear') )
                ])
svr.fit(x_train, y_train)
#Save the model
joblib.dump(svr, 'SVR_Model.pkl', compress =1)


######Tuning Kernel Ridge
kr = Pipeline([
        ('scaler',scaler),
        ('kernelRidge', KernelRidge())
        ])
test_params = {'kernelRidge__kernel':['linear'], 'kernelRidge__alpha':distribution, \
                'kernelRidge__gamma':distribution, 'kernelRidge__degree':[1,2,3]}
kr_optimized = RandomizedSearchCV(estimator = kr, param_distributions  = test_params, scoring=scoring, cv=cv, n_iter=n_iter)
kr_optimized.fit(x_train, y_train)
#Save the model
joblib.dump(kr_optimized, 'Kernel_Ridge_Model.pkl', compress=1)


#####Tuning Lasso
lasso = Pipeline([
                ('scaler', scaler),
                ('lasso', Lasso())
                ])
test_params = {'lasso__alpha':distribution}
lasso_optimized = RandomizedSearchCV(estimator=lasso, param_distributions=test_params, scoring=scoring, cv=cv, n_iter=n_iter)
lasso_optimized.fit(x_train, y_train)

joblib.dump(lasso_optimized, 'Lasso_Model.pkl', compress=1)

######Tuning Linear Ridge
linear_ridge = Pipeline([
                        ('scaler', scaler),
                        ('Ridge', Ridge())
                        ])
test_params = {'Ridge__alpha':distribution}
linear_ridge_optimized = RandomizedSearchCV(estimator = linear_ridge, param_distributions = test_params, scoring=scoring, cv=cv, n_iter=n_iter)
linear_ridge_optimized.fit(x_train, y_train)
#Save the model
joblib.dump(linear_ridge_optimized, 'Linear_Ridge_Model.pkl', compress=1)


######Tuning Bayesian
bayesian_ridge = Pipeline([
                        ('scaler', scaler),
                        ('BayesianRidge', BayesianRidge())
                        ])
test_params = {'BayesianRidge__alpha_1':[10**-5, 10**-4,0.001,0.01,0.1,1,10],
'BayesianRidge__alpha_2':[10**-5, 10**-4,0.001,0.01,0.1,1,10],
'BayesianRidge__lambda_1': [10**-5, 10**-4,0.001,0.01,0.1,1,10],
'BayesianRidge__lambda_2': [10**-5, 10**-4,0.001,0.01,0.1,1,10] }
bayesian_ridge_optimized = RandomizedSearchCV(estimator = bayesian_ridge, param_distributions = test_params, scoring=scoring, cv=cv, n_iter=n_iter)
bayesian_ridge_optimized.fit(x_train,y_train)
#Save the model
joblib.dump(bayesian_ridge_optimized, 'Bayesian_Ridge_Model.pkl', compress=1)


######Tuning SGD
sgd = Pipeline([
                ('scaler', scaler),
                ('SGD', SGDRegressor())
                ])
#sgd.get_params().keys() to get the params
test_params = {'SGD__alpha':[10**-5, 10**-4,0.001,0.01,0.1,1,10],
'SGD__penalty': ['l2', 'l1', 'elasticnet'],
'SGD__epsilon': [10**-5, 10**-4,0.001,0.01,0.1,1,10],
'SGD__l1_ratio': [0.1, 0.2, 0.5, 0.6, 0.8, 0.9],
'SGD__power_t': [0.1, 0.2, 0.25, 0.5, 0.8, 0.9]}
sgd_optimized = RandomizedSearchCV(estimator = sgd, param_distributions = test_params, scoring=scoring, cv=cv, n_iter=n_iter)
sgd_optimized.fit(x_train, y_train)
#Save the model
joblib.dump(sgd_optimized, 'SGD_Model.pkl', compress=1)


######Tuning Random Forest
random_forest = Pipeline([
                        ('scaler', scaler),
                        ('RandomForestRegressor', RandomForestRegressor())
                        ])
test_params = {'RandomForestRegressor__n_estimators':[1,3,5,10,15,20],
'RandomForestRegressor__max_depth': [1,2,3,4,5,6,8],
'RandomForestRegressor__min_samples_split': [1,2,3],
'RandomForestRegressor__min_samples_leaf': [1,2,3],
'RandomForestRegressor__bootstrap': [True, False]}
random_forest_optimized = RandomizedSearchCV(estimator = random_forest, param_distributions = test_params, scoring=scoring, cv=cv, n_iter=n_iter )
random_forest_optimized.fit(x_train,y_train)
#Save the model
joblib.dump(random_forest_optimized, 'Random_Forest_Model.pkl', compress =1)

###########################################
print '########## TESTING ERRORS ##########'
y_predicted = lr.predict(x_test)
y2 = svr.predict(x_test)
y3 = kr_optimized.predict(x_test)
y4 = lasso_optimized.predict(x_test)
y5 = linear_ridge_optimized.predict(x_test)
y6 = bayesian_ridge_optimized.predict(x_test)
y7 = sgd_optimized.predict(x_test)
y8 = random_forest_optimized.predict(x_test)

print "MAE for Linear Regression:", mean_absolute_error(y_test, y_predicted)
print "MAE for SVR:", mean_absolute_error(y_test, y2)
print "MAE for Kernel Ridge Regression:", mean_absolute_error(y_test, y3)
print "MAE for Lasso Regression:", mean_absolute_error(y_test, y4)
print "MAE for Linear Ridge Regression:", mean_absolute_error(y_test, y5)
print "MAE for Bayesian Ridge Regression:", mean_absolute_error(y_test, y6)
print "MAE for Stochastic Gradient Descent Regression:", mean_absolute_error(y_test, y7)
print "MAE for Random Forest Regression:", mean_absolute_error(y_test, y8)
print "--------------------------------"
print "MSE for Linear Regression", mean_squared_error(y_test, y_predicted)
print "MSE for SVR", mean_squared_error(y_test, y2)
print "MSE for Kernel Ridge Regression", mean_squared_error(y_test, y3)
print "MSE for Lasso Regression:", mean_squared_error(y_test, y4)
print "MSE for Linear Ridge Regression:", mean_squared_error(y_test, y5)
print "MSE for Bayesian Ridge Regression:", mean_squared_error(y_test, y6)
print "MSE for Stochastic Gradient Descent Regression:", mean_squared_error(y_test, y7)
print "MSE for Random Forest Regression:", mean_squared_error(y_test, y8)
print "--------------------------------"
print "R2 for Linear Regression", r2_score(y_test, y_predicted)
print "R2 for SVR", r2_score(y_test, y2)
print "R2 for Kernel Ridge Regression", r2_score(y_test, y3)
print "R2 for Lasso Regression:", r2_score(y_test, y4)
print "R2 for Linear Ridge Regression:", r2_score(y_test, y5)
print "R2 for Bayesian Ridge Regression:", r2_score(y_test, y6)
print "R2 for Stochastic Gradient Descent Regression:", r2_score(y_test, y7)
print "R2 for Random Forest Regression:", r2_score(y_test, y8)

############################################

print '########## TRAINING ERRORS ##########'
y_predicted = lr.predict(x_train)
y2 = svr.predict(x_train)
y3 = kr_optimized.predict(x_train)
y4 = lasso_optimized.predict(x_train)
y5 = linear_ridge_optimized.predict(x_train)
y6 = bayesian_ridge_optimized.predict(x_train)
y7 = sgd_optimized.predict(x_train)
y8 = random_forest_optimized.predict(x_train)

print "MAE for Linear Regression:", mean_absolute_error(y_train, y_predicted)
print "MAE for SVR:", mean_absolute_error(y_train, y2)
print "MAE for Kernel Ridge Regression:", mean_absolute_error(y_train, y3)
print "MAE for Lasso Regression:", mean_absolute_error(y_train, y4)
print "MAE for Linear Ridge Regression:", mean_absolute_error(y_train, y5)
print "MAE for Bayesian Ridge Regression:", mean_absolute_error(y_train, y6)
print "MAE for Stochastic Gradient Descent Regression:", mean_absolute_error(y_train, y7)
print "MAE for Random Forest Regression:", mean_absolute_error(y_train, y8)
print "--------------------------------"
print "MSE for Linear Regression:", mean_squared_error(y_train, y_predicted)
print "MSE for SVR:", mean_squared_error(y_train, y2)
print "MSE for Kernel Ridge Regression:", mean_squared_error(y_train, y3)
print "MSE for Lasso Regression:", mean_squared_error(y_train, y4)
print "MSE for Linear Ridge Regression:", mean_squared_error(y_train, y5)
print "MSE for Bayesian Ridge Regression:", mean_squared_error(y_train, y6)
print "MSE for Stochastic Gradient Descent Regression:", mean_squared_error(y_train, y7)
print "MSE for Random Forest Regression:", mean_squared_error(y_train, y8)
print "--------------------------------"
print "R2 for Linear Regression:", r2_score(y_train, y_predicted)
print "R2 for SVR:", r2_score(y_train, y2)
print "R2 for Kernel Ridge Regression:", r2_score(y_train, y3)
print "R2 for Lasso Regression:", r2_score(y_train, y4)
print "R2 for Linear Ridge Regression:", r2_score(y_train, y5)
print "R2 for Bayesian Ridge Regression:", r2_score(y_train, y6)
print "R2 for Stochastic Gradient Descent Regression:", r2_score(y_train, y7)
print "R2 for Random Forest Regression:", r2_score(y_train, y8)
print "--------------------------------"
end_time = time.time()
print "It takes %.2f minutes to train and cross validate with 10 Kfolds and 100 randomized grid searches for all 8 regression models" % ((end_time - start_time)/60)
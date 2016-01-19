import pandas as pd
from sklearn.cross_validation import train_test_split, ShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

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
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.25, random_state =6)

#Scale x to certain range
#scaler = StandardScaler()
scaler = MinMaxScaler(feature_range=(-1,1))
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


#####Tuning Kernel Ridge
kr = KernelRidge()
test_params = {'kernel':['linear', 'rbf'], 'alpha':[10**-7,10**-6,10**-5, 10**-4,0.001,0.01,0.1,1,10], \
                'gamma':[10**-5,10**-4,0.001,0.01,0.1,1,10], 'degree':[1,2,3,4]}
kr_optimized = GridSearchCV(estimator = kr, param_grid = test_params, scoring = 'mean_absolute_error' )
kr_optimized.fit(x_train_scaled, y_train)
y_predicted = kr_optimized.predict(x_test_scaled)

print "Best Parameters for KR: %s" %kr_optimized.best_estimator_
print "MAE for KR:", mean_absolute_error(y_test, y_predicted)
print "MSE for KR", mean_squared_error(y_test, y_predicted)
print "R2 for KR", r2_score(y_test, y_predicted)
'''Best Parameters for KNR: KernelRidge(alpha=0.0001, coef0=1, degree=1, gamma=0.001, kernel='rbf',kernel_params=None)
MAE for KR: 2.35570415904
MSE for KR 20.3426329621
R2 for KR 0.997206503468'''

######Tuning Lasso
lasso = Lasso()
test_params = {'alpha':[10**-9, 10**-8,10**-7,10**-6,10**-5, 10**-4,0.001,0.01,0.1,1,10,100,1000]}
lasso_optimized = GridSearchCV(estimator = lasso, param_grid = test_params, scoring = 'mean_absolute_error' )
lasso_optimized.fit(x_train_scaled, y_train)
y_predicted = lasso_optimized.predict(x_test_scaled)

print "Best Parameters for lasso: %s" %lasso_optimized.best_estimator_
print "MAE for lasso:", mean_absolute_error(y_test, y_predicted)
print "MSE for lasso", mean_squared_error(y_test, y_predicted)
print "R2 for lasso", r2_score(y_test, y_predicted)
'''Best Parameters for lasso: Lasso(alpha=1e-09, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)
MAE for lasso: 3.38177782627
MSE for lasso 41.3155554331
R2 for lasso 0.993617255849'''

######Tuning Linear Ridge
linear_ridge = Ridge()
test_params = {'alpha':[10**-9, 10**-8,10**-7,10**-6,10**-5, 10**-4,0.001,0.01,0.1,1,10,100,1000]}
linear_ridge_optimized = GridSearchCV(estimator = linear_ridge, param_grid = test_params, scoring = 'mean_absolute_error' )
linear_ridge_optimized.fit(x_train_scaled, y_train)
y_predicted = linear_ridge_optimized.predict(x_test_scaled)

print "Best Parameters for linear ridge: %s" %linear_ridge_optimized.best_estimator_
print "MAE for linear ridge:", mean_absolute_error(y_test, y_predicted)
print "MSE for linear ridge", mean_squared_error(y_test, y_predicted)
print "R2 for linear ridge", r2_score(y_test, y_predicted)
'''Best Parameters for linear ridge: Ridge(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=None,
   normalize=False, random_state=None, solver='auto', tol=0.001)
MAE for linear ridge: 3.35795768117
MSE for linear ridge 38.473182419
R2 for linear ridge 0.994056367451'''

######Tuning Bayesian
bayesian_ridge = BayesianRidge()
test_params = {'alpha_1':[10**-5, 10**-4,0.001,0.01,0.1,1,10], \
'alpha_2':[10**-5, 10**-4,0.001,0.01,0.1,1,10], \
'lambda_1': [10**-5, 10**-4,0.001,0.01,0.1,1,10], \
'lambda_2': [10**-5, 10**-4,0.001,0.01,0.1,1,10] }
bayesian_optimized = GridSearchCV(estimator = bayesian_ridge, param_grid = test_params, scoring = 'mean_absolute_error' )
bayesian_optimized.fit(x_train_scaled, y_train)
y_predicted = bayesian_optimized.predict(x_test_scaled)

print "Best Parameters for bayesian: %s" %bayesian_optimized.best_estimator_
print "MAE for bayesian:", mean_absolute_error(y_test, y_predicted)
print "MSE for bayesian", mean_squared_error(y_test, y_predicted)
print "R2 for bayesian", r2_score(y_test, y_predicted)
'''Best Parameters for bayesian: BayesianRidge(alpha_1=1e-05, alpha_2=10, compute_score=False, copy_X=True,
       fit_intercept=True, lambda_1=10, lambda_2=1e-05, n_iter=300,
       normalize=False, tol=0.001, verbose=False)
MAE for bayesian: 3.35807586285
MSE for bayesian 38.4740276071
R2 for bayesian 0.99405623688'''

######Tuning SGD
sgd = SGDRegressor()
test_params = {'alpha':[10**-5, 10**-4,0.001,0.01,0.1,1,10], \
'loss':['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'], \
'penalty': ['none', 'l2', 'l1', 'elasticnet'], \
'epsilon': [10**-5, 10**-4,0.001,0.01,0.1,1,10], \
'l1_ratio': [0.1, 0.2, 0.5, 0.6, 0.8, 0.9], \
'power_t': [0.1, 0.2, 0.25, 0.5, 0.8, 0.9]}
sgd_optimized = GridSearchCV(estimator = sgd, param_grid = test_params, scoring = 'mean_absolute_error' )
sgd_optimized.fit(x_train_scaled, y_train)
y_predicted = sgd_optimized.predict(x_test_scaled)

print "Best Parameters for SGD: %s" %sgd_optimized.best_estimator_
print "MAE for SGD:", mean_absolute_error(y_test, y_predicted)
print "MSE for SGD", mean_squared_error(y_test, y_predicted)
print "R2 for SGD", r2_score(y_test, y_predicted)
'''Best Parameters for SGD: SGDRegressor(alpha=0.1, average=False, epsilon=0.001, eta0=0.01,
       fit_intercept=True, l1_ratio=0.2, learning_rate='invscaling',
       loss='squared_loss', n_iter=5, penalty='none', power_t=0.2,
       random_state=None, shuffle=True, verbose=0, warm_start=False)
#MAE for SGD: 9.04117895779
#MSE for SGD 292.104437304
#R2 for SGD 0.954873464267'''


####Develop models using various tuned algorithms above
lr = LinearRegression()
lr.fit(x_train, y_train)
y_predicted = lr.predict(x_test)

svr = SVR(C=10, gamma =1, kernel = 'linear')
svr.fit(x_train_scaled, y_train)
y2 = svr.predict(x_test_scaled)

kr = KernelRidge(alpha=0.0001, coef0=1, degree=1, gamma=0.001, kernel='rbf',kernel_params=None)
kr.fit(x_train_scaled, y_train)
y3 = kr.predict(x_test_scaled)

lasso = Lasso(alpha=1e-09)
lasso.fit(x_train_scaled, y_train)
y4 = lasso.predict(x_test_scaled)

linear_ridge = Ridge(alpha=0.1)
linear_ridge.fit(x_train_scaled,y_train)
y5 = linear_ridge.predict(x_test_scaled)

bayesian_ridge = BayesianRidge(alpha_1=1e-05, alpha_2=10, lambda_1=10, lambda_2=1e-05)
bayesian_ridge.fit(x_train_scaled, y_train)
y6 = bayesian_ridge.predict(x_test_scaled)

sgd = SGDRegressor(alpha=0.1, epsilon=0.001, l1_ratio=0.2, loss='squared_loss', penalty='none', power_t=0.2)
sgd.fit(x_train_scaled, y_train)
y7 = sgd.predict(x_test_scaled)

###########################################
print '########## TESTING ERRORS ##########'

print "MAE for Linear Regression:", mean_absolute_error(y_test, y_predicted)
print "MAE for SVR:", mean_absolute_error(y_test, y2)
print "MAE for Kernel Ridge Regression:", mean_absolute_error(y_test, y3)
print "MAE for Lasso Regression:", mean_absolute_error(y_test, y4)
print "MAE for Linear Ridge Regression:", mean_absolute_error(y_test, y5)
print "MAE for Bayesian Ridge Regression:", mean_absolute_error(y_test, y6)
print "MAE for Stochastic Gradient Descent Regression:", mean_absolute_error(y_test, y7)
print "--------------------------------"
print "MSE for Linear Regression", mean_squared_error(y_test, y_predicted)
print "MSE for SVR", mean_squared_error(y_test, y2)
print "MSE for Kernel Ridge Regression", mean_squared_error(y_test, y3)
print "MSE for Lasso Regression:", mean_squared_error(y_test, y4)
print "MSE for Linear Ridge Regression:", mean_squared_error(y_test, y5)
print "MSE for Bayesian Ridge Regression:", mean_squared_error(y_test, y6)
print "MSE for Stochastic Gradient Descent Regression:", mean_squared_error(y_test, y7)
print "--------------------------------"
print "R2 for Linear Regression", r2_score(y_test, y_predicted)
print "R2 for SVR", r2_score(y_test, y2)
print "R2 for Kernel Ridge Regression", r2_score(y_test, y3)
print "R2 for Lasso Regression:", r2_score(y_test, y4)
print "R2 for Linear Ridge Regression:", r2_score(y_test, y5)
print "R2 for Bayesian Ridge Regression:", r2_score(y_test, y6)
print "R2 for Stochastic Gradient Descent Regression:", r2_score(y_test, y7)

###########################################
print '########## TRAINING ERRORS ##########'

y_predicted = lr.predict(x_train)
y2 = svr.predict(x_train_scaled)
y3 = kr.predict(x_train_scaled)
y4 = lasso.predict(x_train_scaled)
y5 = linear_ridge.predict(x_train_scaled)
y6 = bayesian_ridge.predict(x_train_scaled)
y7 = sgd.predict(x_train_scaled)

print "MAE for Linear Regression:", mean_absolute_error(y_train, y_predicted)
print "MAE for SVR:", mean_absolute_error(y_train, y2)
print "MAE for Kernel Ridge Regression:", mean_absolute_error(y_train, y3)
print "MAE for Lasso Regression:", mean_absolute_error(y_train, y4)
print "MAE for Linear Ridge Regression:", mean_absolute_error(y_train, y5)
print "MAE for Bayesian Ridge Regression:", mean_absolute_error(y_train, y6)
print "MAE for Stochastic Gradient Descent Regression:", mean_absolute_error(y_train, y7)
print "--------------------------------"
print "MSE for Linear Regression:", mean_squared_error(y_train, y_predicted)
print "MSE for SVR:", mean_squared_error(y_train, y2)
print "MSE for Kernel Ridge Regression:", mean_squared_error(y_train, y3)
print "MSE for Lasso Regression:", mean_squared_error(y_train, y4)
print "MSE for Linear Ridge Regression:", mean_squared_error(y_train, y5)
print "MSE for Bayesian Ridge Regression:", mean_squared_error(y_train, y6)
print "MSE for Stochastic Gradient Descent Regression:", mean_squared_error(y_train, y7)
print "--------------------------------"
print "R2 for Linear Regression:", r2_score(y_train, y_predicted)
print "R2 for SVR:", r2_score(y_train, y2)
print "R2 for Kernel Ridge Regression:", r2_score(y_train, y3)
print "R2 for Lasso Regression:", r2_score(y_train, y4)
print "R2 for Linear Ridge Regression:", r2_score(y_train, y5)
print "R2 for Bayesian Ridge Regression:", r2_score(y_train, y6)
print "R2 for Stochastic Gradient Descent Regression:", r2_score(y_train, y7)
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class combine_models(BaseEstimator, TransformerMixin):

    def __init__(self, model1, model2, model3, model4, model5, model6, model7, model8):
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        self.model5 = model5
        self.model6 = model6
        self.model7 = model7
        self.model8 = model8

    def fit(self, X, y):
        return self

    def transform(self, X):
        predictions = {'model1':[], 'model2': [], 'model3': [], 'model4': [], 'model5':[], 'model6':[], 'model7':[], 'model8':[]}
        predictions['model1'] = list(self.model1.predict(X))
        predictions['model2'] = list(self.model2.predict(X))
        predictions['model3'] = list(self.model3.predict(X))
        predictions['model4'] = list(self.model4.predict(X))
        predictions['model5'] = list(self.model5.predict(X))
        predictions['model6'] = list(self.model6.predict(X))
        predictions['model7'] = list(self.model7.predict(X))
        predictions['model8'] = list(self.model8.predict(X))
        X_train = pd.DataFrame(predictions)
        return X_train
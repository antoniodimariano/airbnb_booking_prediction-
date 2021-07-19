# Created by Antonio Di Mariano (antonio.dimariano@gmail.com) at 15/12/2019
from sklearn.ensemble.forest import RandomForestClassifier
import numpy as np
from xgboost.sklearn import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
import time
class RandomForest:
    def __init__(self):
        self.random_forest = RandomForestClassifier()

    def predict(self, x_train, y_train, x_test):

        try:
            t0 = time.time()
            print("[]Starting RandomForest prediction")
            self.random_forest.fit(x_train, np.ravel(y_train))
            Y_pred = self.random_forest.predict(x_test)
            print("[*] RandomForest Prediction is completed. RF Score:",self.random_forest.score(x_train, y_train))
            print("[*] Time required:",time.time()-t0)
            return Y_pred
        except Exception as error:
            print("ERROR DURING RandomForest:", error)


class XGboost:
    def __init__(self):
        self.clf = XGBClassifier(max_depth=6, learning_rate=0.2, n_estimators=5,
                                 objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)

    def predict(self, x_train, y_train, x_test):

        print("[]Starting XGboost prediction")
        try:
            t0 = time.time()
            self.clf.fit(x_train, y_train)
            Y_pred = self.clf.predict(x_test)
            print("[*] XGBC Prediction is completed. RF Score:", self.clf.score(x_train, y_train))
            print("[*] Time required:", time.time() - t0)
            return Y_pred
        except Exception as error:
            print("ERROR DURING XGboost:",error)
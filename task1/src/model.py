"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2020

Author(s): Anton Loubman, Adam Monsonegro,
Yair Abraham, Eyal Ben Tovim

===================================================
"""
from parser_iml import *
from pandas import DataFrame
from xgboost import XGBRegressor
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
import pickle

FLIGHT_DATA_PATH = "train_data.csv"

class FlightPredictor:
    def __init__(self, path_to_weather=''):
        """
        Initialize an object from this class.
        @param path_to_weather: The path to a csv file containing weather data.
        """
        self.linearReg = XGBRegressor(object='reg:linear', colsample_bytree=0.8, learning_rate=0.5, max_depth=5,
                                      alpha=10, n_estimators=10)
        self.classifier = OneVsRestClassifier(XGBClassifier(n_jobs=-1, max_depth=4, num_class=5, objective="multi:softmax"))
        self.y_hat_arr = None
        self.y_hat_factor = None

    def train(self, path_to_weather):
        """
        This function trains the data
        :param path_to_weather: A path to a csv file containing all the
        weather data
        :return:
        """

        self.parser = CSVParser(path_to_weather, FLIGHT_DATA_PATH)

        # Train models with and without delay
        x_train, y = self.parser.get_train()
        y_train_arr, y_train_delay = y["ArrDelay"], y["DelayFactor"]
        self.linearReg.fit(x_train.values, y_train_arr.values)
        self.classifier.fit(x_train.values, y_train_delay.values)
        pickle_out = open("learner.pickle", "wb")
        median = np.median(y_train_arr.values)
        pickle.dump(self.linearReg, pickle_out)
        pickle.dump(self.classifier, pickle_out)

        pickle.dump(median, pickle_out)

    def predict(self, x):
        """
        Recieves a pandas DataFrame of shape (m, 15) with m flight features, and predicts their
        delay at arrival and the main factor for the delay.
        @param x: A pandas DataFrame with shape (m, 15)
        @return: A pandas DataFrame with shape (m, 2) with your prediction
        """
        pickle_out = open("learner.pickle", "rb")
        self.linearReg = pickle.load(pickle_out)
        self.classifier = pickle.load(pickle_out)
        x["ArrDelay"] = pickle.load(pickle_out)
        x = preProcessing(x)
        x = x.drop(columns=['ArrDelay'])
        self.y_hat_factor = self.classifier.predict(x.values)
        self.y_hat_arr = self.linearReg.predict(x.values)
        return DataFrame({"PredArrDelay": self.y_hat_arr, "PredDelayFactor": self.y_hat_factor})

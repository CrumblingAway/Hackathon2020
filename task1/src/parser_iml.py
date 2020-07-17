"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2020

Author(s): Anton Loubman, Adam Monsonegro,
Yair Abraham, Eyal Ben Tovim

===================================================
"""
import numpy as np
import pandas as pd
import sklearn.model_selection as sk


def preProcessing(df):
    """
    loads the train data, removes irrelevant features and adds new ones
    return the new data
    :param df:
    :return:
    """

    df["year"] = pd.to_datetime(df["FlightDate"], errors="coerce").dt.strftime(
        "%y")
    df["month"] = pd.to_datetime(df["FlightDate"], errors="coerce").dt.strftime(
        "%m")
    df["day"] = pd.to_datetime(df["FlightDate"], errors="coerce").dt.strftime(
        "%d")

    # dropping the categorical
    df = df.drop(columns=['OriginCityName'])
    df = df.drop(columns=['OriginState'])
    df = df.drop(columns=['DestCityName'])
    df = df.drop(columns=['DestState'])
    df = df.drop(columns=['Flight_Number_Reporting_Airline'])
    df = df.drop(columns=['Tail_Number'])
    df = df.drop(columns=["FlightDate"])

    # dealing with relevant categorical features
    df["Reporting_Airline"] = df["Reporting_Airline"].astype("category").cat.codes
    df["Origin"] = df["Origin"].astype("category").cat.codes
    df["Dest"] = df["Dest"].astype("category").cat.codes

    # removing invalid rows
    df = df.loc[(df["CRSDepTime"] >= 0) & (df["CRSDepTime"] <= 2400)]

    df['CRSDepTime'] = (df["CRSDepTime"] // 100) * 60 + df["CRSDepTime"] % 100

    df = df.loc[(df["CRSArrTime"] >= 0) & (df["CRSArrTime"] <= 2400)]

    df['CRSArrTime'] = (df["CRSArrTime"] // 100) * 60 + df["CRSArrTime"] % 100

    df = df.loc[(df["CRSElapsedTime"] > 0)]
    df = df.loc[(df["Distance"] > 0)]

    df["distance_elasped"] = df["Distance"] * df["CRSElapsedTime"]

    df["real_dep"] = df["CRSDepTime"] + df["ArrDelay"]
    df["real_arr"] = df["CRSArrTime"] + df["ArrDelay"]

    df = df.drop(columns=['Distance'])
    df = df.drop(columns=['CRSElapsedTime'])

    return df


class CSVParser:

    """
    a parser class that is responsible to all the parsing and preprocessing needs of our data.
    """
    def __init__(self, path_weather, data_path):
        self.__flight_df = self.load_data(data_path)
        self.__weather_df = self.load_weather(path_weather)
        trainx, self.testx, trainy, self.testy = sk.train_test_split(self.__flight_df[0], self.__flight_df[1])
        self.trainx, self.validatex, self.trainy, self.validatey = sk.train_test_split(trainx, trainy)

    def load_data(self, path):
        """
        loads the train data, removes irrelevant features and adds new ones
        also splits the data to samples and labels
        :param path: A path to the train .csv file
        :return: Two Dataframes
        """
        df = pd.read_csv(path)

        df = preProcessing(df)

        # y values
        df["DelayFactor"] = df["DelayFactor"].astype("category").cat.codes
        y = pd.DataFrame({'ArrDelay': df['ArrDelay'], "DelayFactor": df["DelayFactor"]})
        df = df.drop(columns=['ArrDelay'])
        df = df.drop(columns=['DelayFactor'])
        return df, y

    def load_weather(self, path):

        """
        loads the weather data, removes irrelevant and invalid features
        :param path:
        :return:
        """
        df = pd.read_csv(path, low_memory=False)
        df['day'] = pd.to_datetime(df['day'], errors="coerce").dt.strftime(
            "%y%m%d")  # changing the date format
        df.rename({"day": "FlightDate"})
        # dealing with relevant categorical features
        for i, cat in enumerate(df):
            if i > 1:
                df = df[pd.to_numeric(df[cat], errors='coerce').notnull()]
                df[cat] = df[cat].astype(float)
            df[cat] = np.where((df[cat] == -99), 0, df[cat])
        # removing invalid rows
        df = df.loc[(df["precip_in"] >= 0)]
        df = df.loc[(df["avg_wind_speed_kts"] >= 0)]
        df = df.loc[(df["avg_wind_drct"] >= 0)]
        df = df.loc[(df["min_rh"] >= 0)]
        df = df.loc[(df["avg_rh"] >= 0)]
        df = df.loc[(df["max_rh"] >= 0)]
        df = df.loc[(df["max_wind_speed_kts"] >= 0)]
        df = df.loc[(df["max_wind_gust_kts"] >= 0)]
        df = pd.get_dummies(df, columns=['station'], drop_first=True)
        return df

    def get_train(self):
        """
        :return: return the training x,y
        """
        return self.__flight_df[0], self.__flight_df[1]

    def get_validate(self):
        """
        :return: return the validating x,y
        """
        return self.validatex, self.validatey

    def get_test(self):
        """
        :return: return the testing x,y
        """
        return self.testx, self.testy




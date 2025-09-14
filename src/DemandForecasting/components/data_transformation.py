import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer 
from src.DemandForecasting.exception import CustomException
from src.DemandForecasting.logger import logging
from src.DemandForecasting.utils import save_object
from statsmodels.tsa.stattools import adfuller


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        pass
        # self.data_tranformation_config=DataTransformationConfig()

    def adf_test(self, timeseries, name):
        '''This function is responsible for data transformation'''
        try:
            
            self.timeseries = timeseries
            self.name = name

            print(f'Results of Dickey-Fuller Test: {self.name}')
            dftest = adfuller(self.timeseries, autolag='AIC')
            dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
            for key,value in dftest[4].items():
                dfoutput['Critical Value (%s)'%key] = value
            print(dfoutput['p-value'])
            
            logging.info(f"Results of Dickey-Fuller Test {self.name}:{dfoutput}")

            if dfoutput['p-value'] < 0.005:
                logging.info(f"For {self.name}: The p-value is extremely small (much less than 0.05), so we reject the null hypothesis that the time series is non-stationary.")
                logging.info(f"Therefore, {self.name} can be considered stationary.")
                data_stationary = self.timeseries
                return data_stationary
            else:
                data_stationary = self.timeseries
                return data_stationary
                # return f"{self.name} is not stationary"
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_IT_load_data_path, test_IT_load_data_path, train_IT_solar_generation_data_path, test_IT_solar_generation_data_path):
        try:
            train_IT_load_df=pd.read_csv(train_IT_load_data_path)
            test_IT_load_df=pd.read_csv(test_IT_load_data_path)
            train_IT_solar_generation_df=pd.read_csv(train_IT_solar_generation_data_path)
            test_IT_solar_generation_df=pd.read_csv(test_IT_solar_generation_data_path)

            logging.info("Applying Preprocessing on training and test dataframe: filling nulls")

            # Fill missing values using forward fill
            train_IT_load_df.fillna(method='ffill', inplace=True)
            test_IT_load_df.fillna(method='ffill', inplace=True)
            train_IT_solar_generation_df.fillna(method='ffill', inplace=True) 
            test_IT_solar_generation_df.fillna(method='ffill', inplace=True)


            # Check for missing values again
            logging.info("Missing values after filling nulss:")
            logging.info(f"Sum of missing values of train: {train_IT_load_df.isnull().sum()}")
            logging.info(f"Sum of missing values of test: {test_IT_load_df.isnull().sum()}")
            logging.info(f"Sum of missing values of train: {train_IT_solar_generation_df.isnull().sum()}")
            logging.info(f"Sum of missing values of test: {test_IT_solar_generation_df.isnull().sum()}")

            
            logging.info("ADF testing on train and test data to check wheither they are stationary or not.")
            
            train_IT_load_stationary = self.adf_test(train_IT_load_df, "IT_load_train_data")
            test_IT_load_stationary = self.adf_test(test_IT_load_df, "IT_load_test_data")
            train_IT_solar_generation_stationary = self.adf_test(train_IT_solar_generation_df, "IT_solar_generation_train_data")
            test_IT_solar_generation_stationary = self.adf_test(test_IT_solar_generation_df, "IT_solar_generation_test_data")

            logging.info(f"Completed the preprocessing object")
            # save_object(
            #     file_path=self.data_tranformation_config.preprocessor_obj_file_path,
            #     obj=preprocessing_obj
            # )

            return (
                train_IT_load_stationary,
                test_IT_load_stationary,
                train_IT_solar_generation_stationary,
                test_IT_solar_generation_stationary
            )
        
        except Exception as e:
            raise CustomException(e, sys)
        
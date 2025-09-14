import os
import sys
from src.DemandForecasting.exception import CustomException
from src.DemandForecasting.logger import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.DemandForecasting.utils import read_sql_data
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_IT_load_data_path:str=os.path.join('artifacts', 'train_IT_load.csv')
    test_IT_load_data_path:str=os.path.join('artifacts', 'test_IT_load.csv')
    train_IT_solar_generation_data_path:str=os.path.join('artifacts', 'train_IT_solar_generation.csv')
    test_IT_solar_generation_data_path:str=os.path.join('artifacts', 'test_IT_solar_generation.csv')
    raw_data_path:str=os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            # reading the data from mysql
            # df = read_sql_data()

            # Load the data
            df = pd.read_csv('TimeSeries_TotalSolarGen_and_Load_IT_2016.csv')
            df.head()

            logging.info("Reading data from database")
            os.makedirs(os.path.dirname(self.ingestion_config.train_IT_load_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Split the IT_load data into training and test sets
            train_size = int(len(df['IT_load_new']) * 0.8)
            train_set_IT_load, test_set_IT_load = df['IT_load_new'][:train_size], df['IT_load_new'][train_size:]

            train_set_IT_load.to_csv(self.ingestion_config.train_IT_load_data_path, index=False, header=True)
            test_set_IT_load.to_csv(self.ingestion_config.test_IT_load_data_path, index=False, header=True)

            # Split the data into training and test sets
            train_size = int(len(df['IT_solar_generation']) * 0.8)
            train_set_IT_solar_generation, test_set_IT_solar_generation = df['IT_solar_generation'][:train_size], df['IT_solar_generation'][train_size:]
    
            train_set_IT_solar_generation.to_csv(self.ingestion_config.train_IT_solar_generation_data_path, index=False, header=True)
            test_set_IT_solar_generation.to_csv(self.ingestion_config.test_IT_solar_generation_data_path, index=False, header=True)


            return(
                self.ingestion_config.train_IT_load_data_path,
                self.ingestion_config.test_IT_load_data_path,
                self.ingestion_config.train_IT_solar_generation_data_path,
                self.ingestion_config.test_IT_solar_generation_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
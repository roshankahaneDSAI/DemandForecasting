from src.DemandForecasting.logger import logging
from src.DemandForecasting.exception import CustomException
from src.DemandForecasting.components.data_ingestion import DataIngestion
from src.DemandForecasting.components.data_transformation import DataTransformation
from src.DemandForecasting.components.model_trainer import ModelTrainer
# from src.DemandForecasting.pipelines.prediction_pipeline import CustomData, PredictPipeline

import sys


import pandas as pd
import numpy as np

if __name__=="__main__":
    logging.info("The execution has started.")

    try:
        data_ingestion=DataIngestion()
        train_IT_load_data_path, test_IT_load_data_path, train_IT_solar_generation_data_path, test_IT_solar_generation_data_path=data_ingestion.initiate_data_ingestion()
        print(train_IT_load_data_path)
        print(test_IT_load_data_path)
        print(train_IT_solar_generation_data_path)
        print(test_IT_solar_generation_data_path)

        data_transformation=DataTransformation()
        train_IT_load_stationary, test_IT_load_stationary,train_IT_solar_generation_stationary, test_IT_solar_generation_stationary=data_transformation.initiate_data_transformation(train_IT_load_data_path, test_IT_load_data_path, train_IT_solar_generation_data_path, test_IT_solar_generation_data_path)

        model_trainer=ModelTrainer()
        r2_square = model_trainer.initiate_model_trainer(train_IT_load_stationary, test_IT_load_stationary,train_IT_solar_generation_stationary, test_IT_solar_generation_stationary)
        print("The r2_square is {}".format(r2_square))

    except Exception as e:
        logging.info("Custom exception")
        raise CustomException(e, sys)
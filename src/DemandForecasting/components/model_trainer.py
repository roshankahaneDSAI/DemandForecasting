import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
# from sktime.forecasting.model_selection import (
#                 ExpandingWindowSplitter,
#                 ForecastingGridSearchCV
#             )
# from sktime.forecasting.naive import NaiveForecaster
# from sktime.forecasting.arima import ARIMA
# from sktime.forecasting.compose import MultiplexForecaster
# from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sklearn.metrics import r2_score

from src.DemandForecasting.exception import CustomException
from src.DemandForecasting.logger import logging
from src.DemandForecasting.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_filepath=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def eval_metrics(self, actual, pred):
        rmse=np.sqrt(mean_squared_error(actual, pred))
        mae=mean_absolute_error(actual, pred)
        r2=r2_score(actual, pred)
        return rmse, mae, r2

    def initiate_model_trainer(self, train_IT_load_stationary, test_IT_load_stationary,train_IT_solar_generation_stationary, test_IT_solar_generation_stationary):
        try:
            logging.info(f"Split training and test input data.")
            train_IT_load, test_IT_load, train_IT_solar, test_IT_solar = (
                train_IT_load_stationary,
                test_IT_load_stationary,
                train_IT_solar_generation_stationary,
                test_IT_solar_generation_stationary
            )

            logging.info("Model training on IT load data.")
            # Fit the ARIMA model on train_IT_load data
            model1 = ARIMA(train_IT_load, order=(2,0,2))
            model_fit1 = model1.fit()

            # Make predictions on the test set
            IT_load_preds = model_fit1.predict(start=len(train_IT_load), end=len(train_IT_load)+len(test_IT_load)-1)

            # Calculate RMSE
            (rmse, mae, r2)=self.eval_metrics(test_IT_load, IT_load_preds)
            print(rmse, mae, r2)
            logging.info(f"test_IT_load rmse: {rmse}")
            logging.info(f"test_IT_load r2: {r2}")
            logging.info(f"test_IT_load mae: {mae}")

            logging.info("Model training on IT solar generation data.")
            # Fit the ARIMA model on train_IT_solar_generation data
            model2 = ARIMA(train_IT_solar, order=(2,0,2))
            model_fit2 = model2.fit()

            # Make predictions on the test set
            IT_solar_preds = model_fit2.predict(start=len(train_IT_solar), end=len(train_IT_solar)+len(test_IT_solar)-1)

            # Calculate RMSE
            (rmse, mae, r2)=self.eval_metrics(test_IT_solar, IT_solar_preds)
            print(rmse, mae, r2)
            logging.info(f"test_IT_solar rmse: {rmse}")
            logging.info(f"test_IT_solar r2: {r2}")
            logging.info(f"test_IT_solar mae: {mae}")
            
            return rmse

            # model_report=evaluate_models(X_train, y_train, X_test, y_test, models, params)
            
            # ## To get best model score from dict
            # best_model_score=max(sorted(model_report.values()))

            # ## To get best model score from dict
            # best_model_name=list(model_report.keys())[
            #     list(model_report.values()).index(best_model_score)
            # ]
            # best_model=models[best_model_name]

            # print("This is the best model:")
            # print(best_model_name)

            # model_names=list(params.keys())

            # actual_model=""

            # for model in model_names:
            #     if best_model_name == model:
            #         actual_model=actual_model+model

            # best_params=params[actual_model]

            # mlflow.set_registry_uri("https://dagshub.com/roshankahaneDSAI/mlproject.mlflow")
            # tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme

            # # mlflow

            # import dagshub
            # dagshub.init(repo_owner='roshankahaneDSAI', repo_name='mlproject', mlflow=True)

            # with mlflow.start_run():
            #     predicted_qualities = best_model.predict(X_test)
            #     (rmse, mae, r2)=self.eval_metrics(y_test, predicted_qualities)

            #     mlflow.log_params(best_params)

            #     mlflow.log_metric("rmse", rmse)
            #     mlflow.log_metric("r2", r2)
            #     mlflow.log_metric("mae", mae)

            #     # model registry does not work with file store
            #     if tracking_url_type_store != "file":

            #         # Register the model
            #         # There are other ways to use the model registry, which depends on the
            #         mlflow.sklearn.log_model(best_model, "model")
            #     else:
            #         mlflow.sklearn.log_model(best_model, "model")


            # if best_model_score<0.6:
            #     raise CustomException("No best model found")
            # logging.info(f"Best model found on both training and testing dataset")

            # save_object(
            #     file_path=self.model_trainer_config.trained_model_filepath,
            #     obj=best_model
            # )

            # predicted=best_model.predict(X_test)
            # r2_square=r2_score(y_test, predicted)

            # return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)
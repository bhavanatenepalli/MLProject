import os
import sys
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    AdaBoostRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split


from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import evaluate_model, save_object

@dataclass
class ModelTrainerConfig:
    modeltrainer_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info("split train and test input data")
            X_train,X_test,y_train,y_test=(
                train_arr[:,:-1],
                test_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,-1]
            )
            models={
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBClassifier": XGBRegressor(),
                "k-Neighbors": KNeighborsRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            
            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]

            if best_model_score <0.6:
                raise CustomException("No best model found")
            logging.info("best model not found with score >0.6")
            
            save_object(
                file_path=self.model_trainer_config.modeltrainer_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            r2_score1=r2_score(y_test,predicted)
            return r2_score1

        except Exception as e:
            raise CustomException(e,sys)
    

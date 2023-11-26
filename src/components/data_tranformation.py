import os
import sys

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class TransformationConfig:
    data_transformation_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTranformation:
    def __init__(self):
        self.data_transformation_config=TransformationConfig()
    
    def get_data_transformer_object(self):
        try:

            num_features=['reading_score','writing_score']
            cat_features=['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
            logging.info("Listes all the num and cat features")

            num_pipeline=Pipeline(
                steps=[("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("impute",SimpleImputer(strategy="most_frequent")),
                    ("one-hot-encoding",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("created num and cat pipelines")

            preprocessor=ColumnTransformer(
                transformers=(
                    ("num",num_pipeline,num_features),
                    ("cat",cat_pipeline,cat_features)
                )
            )
            logging.info("returning preprocessor")
            return preprocessor
    
        except Exception as e:
            raise CustomException(e,sys)   

    def initiate_data_transformer(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read train and test data")

            preprocessor_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            num_features=["reading score","writing_score"]

            input_feature_train_data=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_data=train_df[target_column_name]

            input_feature_test_data=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_data=test_df[target_column_name]

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_data)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_data)
            
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_data)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_data)]

            logging.info("applied pre_processing on train, test data")

            save_object(
                file_path=self.data_transformation_config.data_transformation_obj_file_path,
                obj=preprocessor_obj
            )

            return (train_arr,
                    test_arr,
                    self.data_transformation_config.data_transformation_obj_file_path
                    )
           
        except Exception as e:
            raise CustomException(e,sys)


        
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer # Missing values
from sklearn.preprocessing import StandardScaler # Feature scaling 
# from sklearn.preprocessing import OrdinalEncoder # To rank categorical features
# Pipeline
from sklearn.pipeline import Pipeline #To add everything together 
from sklearn.compose import ColumnTransformer # Combine everything together

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging
import os



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        '''this function is responsible for data transformation'''
        try:
            numerical_columns = ['Location', 
                                'MinTemp', 
                                'MaxTemp', 
                                'Rainfall', 
                                'Evaporation', 
                                'Sunshine',
                                'WindGustDir',
                                'WindGustSpeed',
                                'WindDir9am',
                                'WindDir3pm',
                                'WindSpeed9am',
                                'WindSpeed3pm',
                                'Humidity9am',
                                'Humidity3pm',
                                'Pressure9am',
                                'Cloud9am',
                                'Cloud3pm'
                                ]
            categorical_columns = []
            num_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ('scalar',StandardScaler())

            ])
            cat_pipeline=Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')), 
                ('scaler', StandardScaler())
            ])

            logging.info(f"In data_transformation line 60 Categorical Columns: {categorical_columns}")
            logging.info(f"In data_transformation line61 Numerical Columns:{numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]

            )
            return preprocessor
            

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            logging.info(f"Train dataset read in the initiate_data_transformation line 79: {train_df.head()}")
            test_df=pd.read_csv(test_path)
            logging.info(f"Test dataset read in the initiate_data_transformation line 81 {test_df.head()}")

            logging.info("Reading the train and test file in data_transformation")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="RainTomorrow"

            ## divide the train dataset to independent and dependent feature

            input_features_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            ## divide the test dataset to independent and dependent feature

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Test and Train dataset successfully divided into input and target dataset in data_transformation")

            logging.info("Applying Preprocessing on training and test dataframe in data_transformation")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)


            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object in data_transformation")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info(f'Preprocessing obj saved in line 114 in data_transformation: \n{preprocessing_obj}')
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(sys,e)
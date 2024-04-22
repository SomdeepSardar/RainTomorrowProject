import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import pymysql

import pickle
import numpy as np

load_dotenv()

host=os.getenv("host")
user=os.getenv("user")
password=os.getenv("password")
db=os.getenv('db')



def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        mydb=pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connection Established",mydb)
        df=pd.read_sql_query('Select * from weatherdata',mydb)
        logging.info('Reading from Database complete inside utils.py')
        print("Inside utils.py \n")
        print(df.head())

        return df



    except Exception as ex:
        raise CustomException(ex)
    
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def model_evaluation(true, predicted):
    c_mat = confusion_matrix(true, predicted)
    a_score = accuracy_score(true, predicted)
    c_report = classification_report(true, predicted)
    return c_mat, a_score, c_report

def model_performance(X_train, y_train, X_test, y_test, models): 
    try: 
        report = {}
        for i in range(len(models)): 
            model = list(models.values())[i]
            # Train models
            model.fit(X_train, y_train)
            # Test data
            y_test_pred = model.predict(X_test)

            c_mat, a_score, c_report = model_evaluation(y_test, y_test_pred)
            #Accuracy Score 
            print('Model name: ', model)
            print('Model Training Performance')
            print("c_mat\n", c_mat)
            print("a_score in percentage: ", a_score*100)
            print("c_report: ", c_report)

            logging.info(f'Model: , {model}')
            logging.info(f'Model Training Performance')
            logging.info(f"c_mat\n, {c_mat}")
            logging.info(f"a_score, {a_score}")
            logging.info(f"c_report, {c_report}")
            test_model_score = accuracy_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report

    except Exception as e: 
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
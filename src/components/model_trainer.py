import os
import sys
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from src.utils import save_object, model_evaluation, model_performance

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, model_performance


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    # def model_evaluation(self, true, predicted):
    #     c_mat = confusion_matrix(true, predicted)
    #     a_score = accuracy_score(true, predicted)
    #     c_report = classification_report(true, predicted)
    #     return c_mat, a_score, c_report
    
    logging.info("model_evaluation func in model_trainer executed")

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data in model_trainer.py inside initiate_model_trainer class started")
            
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info("data splitting completed successfully in model_trainer line 42")
            logging.info(f"train_array:{X_train}")
            logging.info(f"train_array:{y_train}")
            logging.info(f"train_array:{X_test}")
            logging.info(f"train_array:{y_test}")

            models = {
                'LogisticRegression': LogisticRegression(),
                'KNeighbours': KNeighborsClassifier(n_neighbors=8),
                'RandomForest': RandomForestClassifier()
            }

            model_report: dict = model_performance(X_train, y_train, X_test, y_test, models)

            print(model_report)
            print("\n"*5)
            logging.info(f"Model Report: {model_report}")

            # Best model
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)] 
            
            best_model = models[best_model_name]

            print(f"The best model is {best_model_name}, with Accuracy Score: {best_model_score}")
            print("\n"*10)
            logging.info(f"The best model is {best_model_name}, with Accuracy Score: {best_model_score}")

            save_object(file_path= self.model_trainer_config.trained_model_file_path, obj = best_model)


        except Exception as e: 
            logging.info("Error occured during model training")
            raise CustomException(e,sys)
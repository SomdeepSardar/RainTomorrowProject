import sys 
import os 
from src.exception import CustomException 
from src.logger import logging 
from src.utils import load_object
import pandas as pd

class PredictPipeline: 
    def __init__(self) -> None:
        pass

    def predict(self, features): 
        try: 
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)


            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)

            return pred
        
        except Exception as e: 
            logging.info("Error occured in predict function in prediction_pipeline location")
            raise CustomException(e,sys)



class CustomData:
        def __init__(self, 
                     Location:int,
                     MinTemp:float, 
                     MaxTemp:float, 
                     Rainfall:float, 
                     Evaporation:float, 
                     Sunshine:float, 
                     WindGustDir:int, 
                     WindGustSpeed:float,
                     WindDir9am:int, 
                     WindDir3pm:int,
                     WindSpeed9am:float,
                     WindSpeed3pm: float,
                     Humidity9am:float,
                     Humidity3pm: float,
                     Pressure9am:float,
                     Cloud9am:float,
                     Cloud3pm:float):
            self.Location = Location
            self.MinTemp = MinTemp
            self.MaxTemp = MaxTemp
            self.Rainfall = Rainfall
            self.Evaporation = Evaporation
            self.Sunshine = Sunshine
            self.WindGustDir = WindGustDir 
            self.WindGustSpeed = WindGustSpeed
            self.WindDir9am = WindDir9am
            self.WindDir3pm = WindDir3pm
            self.WindSpeed9am = WindSpeed9am
            self.WindSpeed3pm= WindSpeed3pm
            self.Humidity9am = Humidity9am
            self.Humidity3pm = Humidity3pm
            self.Pressure9am = Pressure9am
            self.Cloud9am = Cloud9am
            self.Cloud3pm = Cloud3pm

            # Function to return the custom data
        def get_data_as_dataframe(self):
            try: 
                custom_data_input_dict = {
                    'Location': [self.Location],
                    'MinTemp': [self.MinTemp], 
                    'MaxTemp': [self.MaxTemp], 
                    'Rainfall': [self.Rainfall], 
                    'Evaporation': [self.Evaporation],
                    'Sunshine': [self.Sunshine],
                    'WindGustDir' : [self.WindGustDir],
                    'WindGustSpeed': [self.WindGustSpeed],
                    'WindDir9am' : [self.WindDir9am],
                    'WindDir3pm' : [self.WindDir3pm], 
                    'WindSpeed9am': [self.WindSpeed9am],
                    'WindSpeed3pm': [self.WindSpeed3pm],
                    'Humidity9am': [self.Humidity9am],  
                    'Humidity3pm': [self.Humidity3pm], 
                    'Pressure9am': [self.Pressure9am],  
                    'Cloud9am': [self.Cloud9am],  
                    'Cloud3pm': [self.Cloud3pm]
                }


                df = pd.DataFrame(custom_data_input_dict)

                logging.info("Dataframe created inside predict_pipeline")
                
                logging.info(df.head())

                return df
            
            except Exception as e:
                logging.info("Error occured in get_data_as_dataframe function in prediction_pipeline")
                raise CustomException(e,sys) 
             
             
        
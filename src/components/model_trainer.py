import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import saved_obj,load_obj
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from sklearn.svm import SVR
from src.utils import saved_obj,evaluate_metrics,load_obj

@dataclass

class ModelTrainerConfig:
    Model_trainer_path_obj:str = os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]
            
            ## initiate model
            models = {
                "LinearRegressor":LinearRegression(),
                "Ridge":Ridge(),
                "Lasso":Lasso(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "SupportVectorRegressor":SVR()
            }
            
            params ={
                "LinearRegressor":{},
                "Ridge":{
                    'alpha':[0.1,1.0,0.2]
                    },
                "Lasso":{
                    'alpha':[0.1,1.0,0.2]
                },
                "DecisionTreeRegressor":{
                    'max_depth':[3,5,None],
                    'min_samples_split':[2],
                    'random_state':[None]
                },
                'RandomForestRegressor':{
                    'n_estimators':[5,10,20],
                    'max_depth':[2,5,10],
                    'min_samples_split':[2]
                },
                'AdaBoostRegressor':{
                    'n_estimators':[50,100,200],
                    'learning_rate':[0.1,0.2,1],
                    'loss':['linear']
                },
                'GradientBoostingRegressor':{
                    'n_estimators':[50,100,200],
                    'learning_rate':[0.1,0.2,1.0],
                    'max_depth':[3,5,10]
                },
                'SupportVectorRegressor':{
                    "kernel": ["linear", "rbf"],
                    "C": [0.1, 1, 10],
                    "gamma": ["scale", "auto"]
                    }}
                
            report,best_model, best_model_name,best_score = evaluate_metrics(X_train,y_train,X_test,y_test,models,params)
                
            logging.info(f"best model found:{best_model_name}: best score: {best_score}")
                
            # saved model
            saved_obj(
                file_path=self.model_trainer_config.Model_trainer_path_obj,
                obj=best_model)
            return report, best_model, best_model_name, best_score
                
            
        except Exception as e:
            raise CustomException(e,sys)
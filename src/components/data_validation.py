import sys
import os
from src.logger import logging
from src.exception import CustomException
import pandas as pd


class Data_Validation:
    def __init__(self,train_data,test_data):
        self.train_data = train_data
        self.test_data = test_data
        
    def initiate_data_validation(self):
        try:
            train_df = pd.read_csv(self.train_data)
            test_df = pd.read_csv(self.test_data)
            logging.info("Read data as data frame")
            
            # shape of data
            logging.info(f"train data shape:\n{train_df.shape}")
            logging.info(f"test data shape: \n{test_df.shape}")
            
            # checking null value
            logging.info(f"train data null value:\n{train_df.isnull().sum()}")
            logging.info(f"test data null value:\n{test_df.isnull().sum()}")
            
            # # data information
            # logging.info(f"train data info: \n{train_df.info()}")
            # logging.info(f"test data info: \n{test_df.info()}")
            
            # checking duplicated
            logging.info(f"train data duplicated:\n{train_df.duplicated().sum()}")
            logging.info(f"test data duplicated:\n{test_df.duplicated().sum()}")
            
            # statatics information
            logging.info(f"train data statatics info:\n{train_df.describe()}")
            logging.info(f"test data statatics info: \n{test_df.describe()}")
            
            expected_columns = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course',
                                'reading_score','writing_score']
            # Check train data
            missing_train_columns = [col for col in expected_columns if col not in train_df.columns]
            if missing_train_columns:
                raise CustomException(f"Missing columns in train data: {missing_train_columns}", sys)
            # Check test data
            missing_test_columns = [col for col in expected_columns if col not in test_df.columns]
            if missing_test_columns:
                raise CustomException(f"Missing columns in test data: {missing_test_columns}", sys)

            
            

        except Exception as e:
            raise CustomException(e,sys)
from src.logger import logging
from src.exception import CustomException
import sys
import os
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import Data_Validation
from src.components.model_trainer import ModelTrainer
def main():
    try:
        
        # step:1 Data Ingestion
        data_ingestion = DataIngestion()
        train_data,test_data = data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion Completed")
        
        # Step: 2 Data Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
        logging.info(f"data tranformation sucessfully")
        
        # step: 3 Data validation
        data_validate = Data_Validation(train_data,test_data)
        data_validate.initiate_data_validation()
        logging.info("data validated sucessfully")
        
        # step : 4 model trainer
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(train_arr,test_arr)
        logging.info("model trained sucessfully")
        
        
        
    except Exception as e:
        raise CustomException(e,sys)
    
    
    
    
if __name__=="__main__":
    main()





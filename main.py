from src.logger import logging
from src.exception import CustomException
import sys
import os
from src.components.data_ingestion import DataIngestion

def main():
    try:
        data_ingestion = DataIngestion()
        train_data,test_dataa = data_ingestion.initiate_data_ingestion()
    except Exception as e:
        raise CustomException(e,sys)
    
    
    
    
if __name__=="__main__":
    main()





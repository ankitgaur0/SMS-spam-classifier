import os,sys
import pickle
import pandas as pd

from src.Exception_Handler import Custom_Exception
from src.Logger import logging
from dataclasses import dataclass

project_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_data_path=os.path.join(project_dir,"Data","extracted_data","spam.csv")
@dataclass
class DataIngest_Config:
    raw_data_path=os.path.join(project_dir,"Data","raw_data.csv")

class Data_Ingest:
    def __init__(self):
        self.data_config_obj=DataIngest_Config()

    def initiate_dataingest(self) ->pd.DataFrame:
        try:
            logging.info("reading the data in the from of pandas DataFrame")
            #reading the data 
            if os.path.exists(input_data_path):
                dataframe=pd.read_csv(input_data_path,encoding="latin")
            else:
                print("provide the existing path.")
                raise Custom_Exception(FileNotFoundError,sys)
            
            # save the dataframe in data Folder as raw_data
            os.makedirs(os.path.dirname(self.data_config_obj.raw_data_path),exist_ok=True)
            with open(self.data_config_obj.raw_data_path,"w") as file_refernce:
                dataframe.to_csv(file_refernce)

            logging.info("save the dataframe data as raw_data.csv complete")

            # return the data frame
            return dataframe



        except Exception as e:
            raise Custom_Exception(e,sys)
        




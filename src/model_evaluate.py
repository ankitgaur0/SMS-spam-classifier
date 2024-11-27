import os,sys
import pandas as pd
from dataclasses import dataclass
from src.Logger import logging
from src.Exception_Handler import Custom_Exception


# the project directory path
project_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
@dataclass
class Data_Config:
    transformed_data_path :str=os.path.join(project_dir,"Data","transformed_data.pickle")
    tfdf_obj :str = os.path.join(project_dir,"artifacts","vectorization.pickle")


class Models_Evaluate:
    def __init__(self):
        self.config_obj=Data_Config()
    
    def train_model(self):
        pass
    def initiate_evaluation(self):
        pass
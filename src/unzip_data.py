import os,sys
import zipfile
from src.Exception_Handler import Custom_Exception
from src.Logger import logging


project_directory=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
zip_file_data=os.path.join(project_directory,"Data","archive.zip")
output_path=os.path.join(project_directory,"Data","extracted_data")

class Unzip_Data:
    def __init__(self,zip_file_path:str ):
        self.zip_file_path=zip_file_data
    
    def initiate_unzip(self):
        try:
            if self.zip_file_path =="":
                print("provide the existing file path for unzip")
                raise FileNotFoundError
            else:
                with zipfile.ZipFile(self.zip_file_path,"r") as file:
                    file.extractall(output_path)
                logging.info("Unzip the file is completed.")


        except Exception as e:
            raise Custom_Exception(e,sys)
        


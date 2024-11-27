import os,sys
import pandas as pd
from src.Logger import logging
from src.Exception_Handler import Custom_Exception


project_directory=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
zip_file_data=os.path.join(project_directory,"Data","archive.zip")
try:
    #call the unzip data file to initiate.
    from src.unzip_data import Unzip_Data
    unzip_obj=Unzip_Data(zip_file_data)
    unzip_obj.initiate_unzip()


    # Perform the Data_Ingestion part
    from src.Data_Ingestion import Data_Ingest
    data_ingest_obj=Data_Ingest()
    dataframe:pd.DataFrame=data_ingest_obj.initiate_dataingest()


    # Perform the Data_preprocessing part
    from src.Data_preprocessing import Data_preprocessing
    preprocessor_obj=Data_preprocessing()
    X,y=preprocessor_obj.initiate_transformation()

    # vectorization the text transformed feature with Text_ectorization.py file script
    from src.Text_vectorization import Text_vectorization
    vectorization=Text_vectorization()
    dataframe=vectorization.initiate_vectorization()

except Exception as e:
    raise Custom_Exception(e,sys)
import os,sys
import pandas as pd
import numpy as np
import pickle
from dataclasses import dataclass
from src.Logger import logging
from src.Exception_Handler import Custom_Exception

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


# the project directory path
project_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# input data path (independant_data.csv) store the transformed_text feature
input_data_path=os.path.join(project_dir,"Data","independant_data.csv")

@dataclass
class Data_Config:
    transformed_data_path :str=os.path.join(project_dir,"Data","transformed_data.pickle")
    CV_obj:str =os.path.join(project_dir,"artifacts","Count_Vector.pickle")
    tfidf_obj :str = os.path.join(project_dir,"artifacts","tfidf.pickle")
    tfidf_obj_for1000 :str = os.path.join(project_dir,"artifacts","tfidf_maxfeat_1000.pickle")
    tfidf_obj_for2000 :str = os.path.join(project_dir,"artifacts","tfidf_maxfeat_2000.pickle")
    tfidf_obj_for3000 :str = os.path.join(project_dir,"artifacts","tfidf_maxfeat_3000.pickle")



class Text_vectorization:
    def __init__(self):
        self.config_obj=Data_Config()
    

    def initiate_vectorization(self):
        try:
            # read the input data file
            dataframe=pd.read_csv(input_data_path)

            # makes the object of CountVectorizer and tfidVectorizer
            cvectors=CountVectorizer()
            tfidf=TfidfVectorizer()
            tfidf_max_feature_1000=TfidfVectorizer(max_features=1000)
            tfidf_max_feature_2000=TfidfVectorizer(max_features=2000)
            tfidf_max_feature_3000=TfidfVectorizer(max_features=3000)

            # now fit_transformed with transformed_feature in data
            dataframe["CountVectors"]=list[(cvectors.fit_transform(dataframe["transformed_text"]).toarray())]
            dataframe["tfidf_vectors"]=list[(tfidf.fit_transform(dataframe["transformed_text"]).toarray())]
            dataframe["tfidf_max_feat.1000"]=list[(tfidf_max_feature_1000.fit_transform(dataframe["transformed_text"]).toarray())]
            dataframe["tfidf_max_feat.2000"]=list[(tfidf_max_feature_2000.fit_transform(dataframe["transformed_text"]).toarray())]
            dataframe["tfidf_max_feat.3000"]=list[(tfidf_max_feature_3000.fit_transform(dataframe["transformed_text"]).toarray())]

            # now store the independant(transformed independant_data.pickle)
            with open(self.config_obj.transformed_data_path,"wb") as path_ref:
                pickle.dump(path_ref,dataframe)

            # now save  the vectorization obj for predication pipeline.
            os.makedirs(self.config_obj.CV_obj,exist_ok=True)
            with open(self.config_obj.CV_obj,"wb") as path_ref:
                pickle.dump(path_ref,cvectors)

            with open(self.config_obj.tfidf_obj) as path_ref:
                pickle.dump(path_ref,tfidf)
            
            with open(self.config_obj.tfidf_obj_for1000) as path_ref:
                pickle.dump(path_ref,tfidf_max_feature_1000)

            with open(self.config_obj.tfidf_obj_for2000) as path_ref:
                pickle.dump(path_ref,tfidf_max_feature_2000)

            with open(self.config_obj.tfidf_obj_for3000) as path_ref:
                pickle.dump(path_ref,tfidf_max_feature_3000)


            # now return the dataframe(transformed dataframe with vectorization features)
            return dataframe
        except Exception as e:
            raise Custom_Exception(e,sys)
        
    

if __name__=="__main__":
    obj=Text_vectorization()
    dataframe=obj.initiate_vectorization()
    print(dataframe.head(5))
    print("+"* 40)
    print(dataframe.isnull().sum())
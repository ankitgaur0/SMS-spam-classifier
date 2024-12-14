import os,sys
import pickle
import pandas as pd
from dataclasses import dataclass

from src.Exception_Handler import Custom_Exception
from src.Logger import logging
from sklearn.preprocessing import LabelEncoder
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string



project_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_file_path=os.path.join(project_dir,"Data","raw_data.csv")
# making the object of nltk.PorterStemmer
ps=PorterStemmer()


@dataclass
class Data_path_Config:
    independant_data_path:str =os.path.join(project_dir,"Data","independant_data.csv")
    dependant_data_path :str= os.path.join(project_dir,"Data","dependant_data_path.csv")

class Data_preprocessing:
    def __init__(self):
        self.Data_config_obj=Data_path_Config()
    

    def transform_text(self,text : str):
        try:
            logging.info("transformed the text feature entries w.r.t. nlp process")
            # convert the text to lower case
            text=text.lower()
            #convert the text into tokenization means break into words.
            text=nltk.word_tokenize(text)
            # removing the special character , whereas now text is a list.
            text_list=[]
            for i in text:
                if i.isalnum():
                    text_list.append(i)
            text=text_list[:]
            text_list.clear()

            #Now removing stop words and punctuation
            for i in text:
                if i not in stopwords.words('english') and i not in string.punctuation:
                    text_list.append(i)

            # reassign the text using the text_list[:]
            text=text_list[:]
            text_list.clear()
            for i in text:
                text_list.append(ps.stem(i))

            # Now we have to join this with string to make string.
            transformed_text=" ".join(text_list)
            logging.info(f"Original: {text}, Transformed: {transformed_text}")
            return transformed_text
        except Exception as e:

            logging.error(f"Error processing text: {text}")
            return ""  # Default to empty string on error

            raise Custom_Exception(e,sys)


    def initiate_transformation(self):
        try:
            # get the data
            if os.path.exists(input_file_path):
                dataframe=pd.read_csv(input_file_path)
            else:
                print("provide the existing path")
                raise Custom_Exception(FileNotFoundError,sys)
            

            # Drop the unnecessary feature in the data frame.
            dataframe.drop(columns=["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1,inplace=True)
            #Renaming the columns names to identification easly.
            dataframe.rename(columns={"v1":"target","v2":"text"},inplace=True)

            # drop the duplicated entries.
            dataframe=dataframe.drop_duplicates(keep="first")

            #make new column of number of character used in of text feature(v2) with respect to each row
            dataframe["no_char"]=dataframe["text"].apply(len)

           
            # make the new feature of transformed_text
            dataframe["transformed_text"]=dataframe["text"].apply(self.transform_text)
            # remove the null values in transformed entries (and in entire data)
            dataframe=dataframe.dropna()

            # separate the independant and dependant features(means X or y)
            X=dataframe.drop(columns="target",axis=1)
            y=dataframe["target"].values

            # encode the y values in the form of binary(0,1)
            encoder=LabelEncoder()
            y=encoder.fit_transform(y)

            #store the transformed data into csv file for further use.
            with open(self.Data_config_obj.independant_data_path,"wb") as file_reference:
                X.to_csv(file_reference,index=False)
            with open(self.Data_config_obj.dependant_data_path,"wb") as file_reference:
                pd.DataFrame(y,columns=["target"]).to_csv(file_reference,index=False)
            # now return the dataframe
            return (
                X,
                y
            )


        except Exception as e:
            raise Custom_Exception(e,sys)
        

if __name__=="__main__":
    obj=Data_preprocessing()
    X,y=obj.initiate_transformation()
    print(X.isnull().sum())

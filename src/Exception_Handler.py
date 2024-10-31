import os,sys
from pathlib import Path


class Custom_Exception(Exception):
    def __init__(self,error,error_details:sys):
        self.error_message=error
        _,_,exc_tb=error_details.exc_info()

        # iterate the linenumber and file name from the executeable information.
        self.file_Name=exc_tb.tb_frame.f_code.co_filename
        self.line_number=exc_tb.tb_lineno


    #Now print the these with the help of __str__ function
    def __str__(self) -> str:
        return f" \n the file name is :{self.file_Name} \n error line number is : {self.line_number} \n error : {str(self.error_message)}"
    

if __name__=="__main__":
    try:
        a=4
        b=0
        print(a/b)
    except Exception as e:
        raise Custom_Exception(e,sys)    
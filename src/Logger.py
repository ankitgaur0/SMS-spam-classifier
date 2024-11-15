import os,sys
from datetime import datetime
import logging


log_name=f"{datetime.now().strftime('%H:%M:%S -%Y-%m-%d')}.log"
project_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#make the full path.
log_path=os.path.join(project_dir,"Logs",log_name)
os.makedirs(log_path,exist_ok=True)

file_name=os.path.join(log_path,log_name)
log_format="[%(asctime)s] %(levelname)s -%(name)s -%(filename)s :%(lineno)d -%(message)s"

#define the basic configuration of the logging.
logging.basicConfig(
    filename=file_name,
    level=logging.INFO,
    format=log_format
)
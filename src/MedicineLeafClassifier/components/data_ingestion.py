import os
import urllib.request as request 
import zipfile
from MedicineLeafClassifier import logger
from MedicineLeafClassifier.utils.common import get_size
from MedicineLeafClassifier.entity.config_entity import DataIngestionConfig
import opendatasets as od
from pathlib import Path 

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if len(os.listdir(self.config.unzip_dir))==0:
            od.download(
                self.config.source_url,
                data_dir = self.config.root_dir
                )
            # filename,headers = request.urlretrieve(
            #     url = self.config.source_url,
            #     filename = self.config.local_data_file
            # )
           # logger.info(f"{self.config.local_data_file} download with following info: {headers}")
            logger.info(f"Data downloaded at {self.config.root_dir}")
        else:
            logger.info("Number of folders exist :" + str(len(os.listdir("./artifacts/data_ingestion/indian-medicinal-leaf-image-dataset/Medicinal Leaf dataset"))))
            # logger.info(f"f already exists of size: {get_size(Path(self.config.local_data_file))}")

    # def extract_zip_file(self):
    #     """
    #     zip_file_path: str
    #     Extracts the zip file into the data directory
    #     Function returns None
        
    #     """
    #     unzip_path = self.config.unzip_dir
    #     os.makedirs(unzip_path,exist_ok=True)
    #     with zipfile.ZipFile(self.config.local_data_file,'r') as zip_ref:
    #         zip_ref.extractall(unzip_path)

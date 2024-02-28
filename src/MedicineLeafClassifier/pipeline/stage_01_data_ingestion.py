from MedicineLeafClassifier.config.configuration import ConfigurationManager
from MedicineLeafClassifier.components.data_ingestion import DataIngestion
from MedicineLeafClassifier import logger

STAGE_NAME = "Data Ingestion Stage"

class DataingestionTrainigPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()


if __name__ == "__main__":
    try:
        logger.info("f>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<")
        obj = DataingestionTrainigPipeline()
        obj.main()
        logger.info(f">>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<\n\nx=================X")
    except Exception as e:
        logger.exception(e)
        raise e
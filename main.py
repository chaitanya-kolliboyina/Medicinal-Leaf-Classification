from MedicineLeafClassifier import  logger
from MedicineLeafClassifier.pipeline.stage_01_data_ingestion import DataingestionTrainigPipeline

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info("f>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<")
    obj = DataingestionTrainigPipeline()
    obj.main()
    logger.info(f">>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<\n\nx=================X")
except Exception as e:
    logger.exception(e)
    raise e
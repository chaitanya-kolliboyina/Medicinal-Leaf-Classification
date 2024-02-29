from MedicineLeafClassifier import  logger
from MedicineLeafClassifier.pipeline.stage_01_data_ingestion import DataingestionTrainigPipeline
from MedicineLeafClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from MedicineLeafClassifier.pipeline.stage_03_training import ModelTrainingPipeline
from MedicineLeafClassifier.pipeline.stage_04_evaluation import EvaluationPipeline
STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info("f>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<")
    obj = DataingestionTrainigPipeline()
    obj.main()
    logger.info(f">>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<\n\nx=================X")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Prepare Base Model"

try:
    logger.info(f"*****************")
    logger.info(f">>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<\n\nx=================X")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Training"

try:
    logger.info(f"*********************")
    logger.info(f">>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<")
    model_trainer = ModelTrainingPipeline()
    model_trainer.main()
    logger.info(f">>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<\n\nx=================X")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Evaluation"

try:
    logger.info(f"***************")
    logger.info(f">>>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<<")
    obj = EvaluationPipeline()
    obj.main()
    logger.info(f">>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
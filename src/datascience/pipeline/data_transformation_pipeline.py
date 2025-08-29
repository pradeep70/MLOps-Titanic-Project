from datascience.config.configuration import ConfigurationManager
from datascience.components.data_transformation import DataTransformation
from datascience import logger


from pathlib import Path


STAGE_NAME="Data Transformation Stage"
logger.info("Logging has started for the stage %s" %STAGE_NAME)

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_transformation(self):
        logger.info("Initiating data transformation")

        try:
            with open(Path("artifacts/data_validation/status.txt"),'r') as f:
                status=f.read().split(" ")[-1]
                logger.info(f"Status of data validation is {status}")
            if status=="True":
                config=ConfigurationManager()
                data_transformation_config=config.get_data_transformation_config()
                data_transformation=DataTransformation(config=data_transformation_config)
                data_transformation.train_test_splitting()
            else:
                raise Exception("Your data scheme is not valid")
            
        except Exception as e:
            print(e)


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.initiate_data_transformation()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
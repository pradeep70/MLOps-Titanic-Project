import os
from src.datascience import logger
from sklearn.model_selection import train_test_split
from src.datascience.entity.config_entity import DataTransformationConfig
import pandas as pd

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config



    def data_transformation(self,titanic):
        titanic['Age'] = titanic['Age'].fillna(titanic['Age'].mean())
        titanic['Cabin'] = titanic['Cabin'].fillna('Unknown')
        titanic['Embarked'] = titanic['Embarked'].fillna('S')

        titanic['Sex'] = titanic['Sex'].map({'male': 0, 'female': 1})
        titanic['Embarked'] = titanic['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
        titanic = titanic.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")
        return titanic

    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    
    def train_test_splitting(self):
        data=pd.read_csv(self.config.data_path)
        data = self.data_transformation(data)

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
from models.database import database
from datetime import datetime

class TrainedModels(database.Model):
    id = database.Column(database.Integer, primary_key = True)
    model_name = database.Column(database.String(100), nullable=False)
    trained_time = database.Column(database.String(200), nullable=False)
    dataset_name = database.Column(database.String(500), nullable=False)
    training_algorithm = database.Column(database.String(100), nullable=False)
    accuracy = database.Column(database.String(100), nullable=False)
    precision = database.Column(database.String(100), nullable=False)
    recall = database.Column(database.String(100), nullable=False)
    f1 = database.Column(database.String(100), nullable=False)
    folderName = database.Column(database.String(600), nullable=False)
    number_of_features = database.Column(database.String(600), nullable=False)


    def __init__(self, model_name, dataset_name, training_alogrithm, accuracy, precision, recall, f1, folderName, number_of_features):
        self.model_name = model_name
        self.trained_time = datetime.now().strftime("%d %B %Y, %I:%M%p")
        self.dataset_name = dataset_name
        self.training_algorithm = training_alogrithm
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.folderName = folderName
        self.number_of_features = number_of_features

    def save(self):
        database.session.add(self)
        database.session.commit()





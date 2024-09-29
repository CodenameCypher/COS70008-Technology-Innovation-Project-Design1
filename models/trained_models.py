from models.database import database


class TrainedModels(database.Model):
    id = database.Column(database.Integer, primary_key = True)
    name = database.Column(database.String(100), nullable=False)
    details = database.Column(database.String(100), nullable=False)


    def __init__(self, name, details):
        self.name = name
        self.details = details

    def save(self):
        database.session.add(self)
        database.session.commit()


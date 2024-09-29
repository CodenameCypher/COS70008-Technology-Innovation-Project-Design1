from models.database import database

import bcrypt

class User(database.Model):
    id = database.Column(database.Integer, primary_key = True)
    name = database.Column(database.String(100), nullable=False)
    email = database.Column(database.String(100), unique=True)
    password = database.Column(database.String(100), nullable=False)
    isAdmin = database.Column(database.Boolean, unique = False, default = False)


    def __init__(self, name, email, password):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))
    
    def save(self):
        database.session.add(self)
        database.session.commit()



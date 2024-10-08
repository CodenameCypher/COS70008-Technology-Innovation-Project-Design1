from flask import Flask, render_template
from controllers import blueprint
from models.database import database

app = Flask(__name__, template_folder="views", static_folder="instance")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.secret_key = "cos70008swinburneuniversityoftechnology"



@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


if __name__ == "__main__":
    database.init_app(app)
    with app.app_context():
        database.create_all()
    app.register_blueprint(blueprint.bp)
    app.run(debug=True)
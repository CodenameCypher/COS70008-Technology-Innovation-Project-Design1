from controllers import blueprint
from controllers.middleware import auth
from flask import render_template, session
from models.trained_models import TrainedModels
from models.user import User

@blueprint.bp.route('/analyze/<string:model_name>')
@auth
def analyze(model_name):
    model_object = TrainedModels.query.filter_by(model_name=model_name).first()
    userObject = User.query.filter_by(id=session['user_id']).first()

    return render_template('admin/analyze.html', model = model_object) if userObject.isAdmin else render_template('user/analyze_model.html', model = model_object)
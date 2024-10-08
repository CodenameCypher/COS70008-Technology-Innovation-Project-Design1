from controllers import blueprint
from controllers.middleware import auth
from flask import render_template
from models.trained_models import TrainedModels

@blueprint.bp.route('/analyze/<string:model_name>')
@auth
def analyze(model_name):
    model_object = TrainedModels.query.filter_by(model_name=model_name).first()

    return render_template('admin/analyze.html', model = model_object)
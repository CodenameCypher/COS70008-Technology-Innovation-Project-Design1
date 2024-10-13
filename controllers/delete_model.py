from controllers import blueprint
from controllers.middleware import admin, auth
from flask import flash, redirect
import os
from app import app
from models.trained_models import TrainedModels
from models.database import database
import shutil

@blueprint.bp.route('/admin/models/delete/<string:model_name>')
@auth
@admin
def delete_model(model_name):
    try:
        shutil.rmtree(os.path.join(app.instance_path, 'models', model_name))
        # os.rmdir(os.path.join(app.instance_path, 'models', model_name))
        TrainedModels.query.filter_by(model_name=model_name).delete()
        database.session.commit()
        flash(model_name+" deleted successfully!",'success')
    except:
        flash("Could not delete "+model_name+"!",'danger')
    return redirect('/admin/train')
    
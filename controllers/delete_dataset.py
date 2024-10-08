from controllers import blueprint
from controllers.middleware import admin, auth
from flask import flash, redirect
import os
from app import app

@blueprint.bp.route('/admin/datasets/delete/<string:file>')
@auth
@admin
def delete_dataset(file):
    try:
        os.remove(os.path.join(app.instance_path, 'datasets',file))
        flash(file+" deleted successfully!",'success')
    except:
        flash("Could not delete "+file+"!",'danger')
    return redirect('/admin/datasets')
    
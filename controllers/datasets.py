from controllers import blueprint
from controllers.middleware import admin, auth
from flask import render_template
import os
from app import app
from datetime import datetime
from hurry.filesize import size


@blueprint.bp.route('/admin/datasets')
@auth
@admin
def datasets():
    file_list = os.listdir(os.path.join(app.instance_path, 'datasets'))
    file_details = []
    for filename in file_list:
        file_created = datetime.fromtimestamp(os.path.getctime(os.path.join(app.instance_path, 'datasets',filename)))
        file_creation_time = file_created.strftime("%d %B %Y, %I:%M%p")
        file_details.append(
            [filename,
            file_creation_time,
            size(os.path.getsize(os.path.join(app.instance_path, 'datasets',filename)))]
        )
    return render_template('admin/datasets.html', files = file_details)
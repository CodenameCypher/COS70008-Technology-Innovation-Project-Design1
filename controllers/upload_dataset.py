from controllers import blueprint
from controllers.middleware import admin, auth
from flask import render_template, request, flash, redirect
from app import app
from werkzeug.utils import secure_filename
import os

@blueprint.bp.route('/admin/datasets/upload', methods=['GET','POST'])
@auth
@admin
def upload_dataset():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect('/admin/datasets/upload')
        if file and allowed_file(file.filename, 'csv'):
            filename = secure_filename(file.filename)
            os.makedirs(os.path.join(app.instance_path, 'datasets'), exist_ok=True)
            file.save(os.path.join(app.instance_path, 'datasets', filename))
            flash('Successfully uploaded '+file.filename, 'success')
            return redirect('/admin/datasets')
        else:
            flash('File must be in .csv format!', 'danger')
            return redirect('/admin/datasets')
    else:
        return render_template('admin/upload_dataset.html')
    
def allowed_file(filename, filetype):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == filetype
from sklearn.calibration import LabelEncoder
from controllers import blueprint
from controllers.middleware import admin, auth
from flask import flash, render_template, request, redirect
import os
import pandas as pd
from app import app
from models.trained_models import TrainedModels

@blueprint.bp.route('/admin/edit/<string:file>', methods = ['GET', 'POST'])
@auth
@admin
def edit_model(file):
    model_object = TrainedModels.query.filter_by(model_name=file).first()
    dataset = pd.read_csv(os.path.join(app.instance_path, 'datasets',model_object.dataset_name))
    dataset = preprocess_data(dataset, model_object.class_name)

    if request.method == 'POST':
        class_names = []

        for i in range(len(dataset[model_object.class_name].unique())):
            class_names.append(request.form[str(dataset[model_object.class_name].unique()[i])])

        dataframe = pd.DataFrame(
            {
                'class_name':class_names,
                'class_label': dataset[model_object.class_name].unique()
            }
        )

        dataframe.to_csv(os.path.join(model_object.folderName,'metadata.csv'), sep=',', index=False)
        flash('Successfully saved class names!', 'success')
        return redirect('/admin/edit/'+model_object.model_name)
    else:
        print(dataset[model_object.class_name].unique())
        try:
            metadata_file = pd.read_csv(os.path.join(model_object.folderName,'metadata.csv'))['class_name']
        except:
            metadata_file = []
        return render_template('/admin/model_edit.html', model = model_object, classes = dataset[model_object.class_name].unique(), class_name = metadata_file)
    
def preprocess_data(dataset, class_column):
    null_values = dataset.isnull().sum()
    
    if null_values.sum() > 0:
        dataset = dataset.dropna()

    for column in dataset.columns:
        if dataset[column].dtype == 'object':
            try:
                dataset[column] = pd.to_numeric(dataset[column], errors='raise')
            except:
                pass
    
    if class_column:
        if dataset[class_column].dtype == 'object':
            label_encoder = LabelEncoder()
            dataset[class_column] = label_encoder.fit_transform(dataset[class_column])
    
    return dataset
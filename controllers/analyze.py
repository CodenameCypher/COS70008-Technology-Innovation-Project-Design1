from sklearn.calibration import LabelEncoder
from controllers import blueprint
from controllers.middleware import auth
from flask import render_template, session
from models.trained_models import TrainedModels
from models.user import User
import pandas as pd
import os
from app import app

@blueprint.bp.route('/analyze/<string:model_name>')
@auth
def analyze(model_name):
    model_object = TrainedModels.query.filter_by(model_name=model_name).first()
    userObject = User.query.filter_by(id=session['user_id']).first()
    dataset = pd.read_csv(os.path.join(app.instance_path, 'datasets',model_object.dataset_name))
    dataset = preprocess_data(dataset, model_object.class_name)
    
    try:
        metadata_file = pd.read_csv(os.path.join(model_object.folderName,'metadata.csv'))
        classes = metadata_file['class_name'].to_list()
        class_labels = metadata_file['class_label'].to_list()
    except:
        classes = ['N/A']*len(dataset[model_object.class_name].unique())
        class_labels = dataset[model_object.class_name].unique()

    return render_template('admin/analyze.html', model = model_object, classes = classes, class_labels = class_labels) if userObject.isAdmin else render_template('user/analyze_model.html', model = model_object , classes = classes, class_labels = class_labels)

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
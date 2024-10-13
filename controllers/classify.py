import pickle
import numpy as np
from controllers import blueprint
from controllers.middleware import auth, user
from flask import render_template, request, redirect
from models.trained_models import TrainedModels
import pandas as pd
import os
from app import app

@blueprint.bp.route('/user/classify/<string:model_name>', methods = ['GET', 'POST'])
@auth
@user
def classify(model_name):
    model_object = TrainedModels.query.filter_by(model_name=model_name).first()
    feature_list = pd.read_csv(os.path.join(app.instance_path, 'models',model_name,'selected_features.csv'))

    if request.method == 'POST':
        inputs = []

        for i in range(feature_list.shape[0]):
            inputs.append(request.form[feature_list['Feature'][i]])
        
        prediction = predict(model_name,os.path.join(app.instance_path, 'models',model_name),inputs)

        try:
            metadata = pd.read_csv(os.path.join(model_object.folderName,'metadata.csv'))['class_name']
            predicted_class = metadata[prediction[0]]
        except:
            predicted_class = prediction[0]

        print(prediction)
        return render_template('user/result.html', model = model_object, prediction = predicted_class)
    else:
        return render_template('user/classify.html', model = model_object, features = feature_list['Feature'])
    

def predict(model_name, folderPath, inputs):
    inputs = np.array(inputs)
    inputs = inputs.reshape(1,-1)

    loaded_scalar = pickle.load(open(os.path.join(folderPath,'scalar.pkl'), 'rb'))
    loaded_model = pickle.load(open(os.path.join(folderPath,'model.pkl'), 'rb'))

    inputs = loaded_scalar.transform(inputs)

    return loaded_model.predict(inputs)
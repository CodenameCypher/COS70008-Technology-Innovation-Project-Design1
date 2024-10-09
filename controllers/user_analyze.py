from controllers import blueprint
from controllers.middleware import auth, user
from flask import render_template, request, redirect, flash
from app import app
import os
from datetime import datetime
from models.trained_models import TrainedModels


@blueprint.bp.route('/user/analyze')
@auth
@user
def user_analyze():
    trained_models = TrainedModels.query.all()

    return render_template('user/analyze.html', models = trained_models)
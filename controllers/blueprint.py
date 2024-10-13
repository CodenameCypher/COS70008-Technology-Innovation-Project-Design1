from flask import Blueprint

bp = Blueprint('main', __name__)

from controllers import login, profile, registration, logout, user_dashboard, admin_dashboard, general_dashboard, datasets, upload_dataset, delete_dataset, manage_users, delete_user, train_analyze, delete_model, analyze, user_analyze, classify, admin_model_edit


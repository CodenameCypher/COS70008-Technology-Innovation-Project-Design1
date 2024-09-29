from flask import Blueprint

bp = Blueprint('main', __name__)

from controllers import login, registration, logout, user_dashboard, admin_dashboard


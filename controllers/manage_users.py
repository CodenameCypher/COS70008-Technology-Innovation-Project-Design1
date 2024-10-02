from controllers import blueprint
from controllers.middleware import admin, auth
from flask import render_template
from app import app
from models.user import User


@blueprint.bp.route('/admin/manage')
@auth
@admin
def manage_users():
    users = User.query.all()
    return render_template('admin/manage_users.html', users = users)
from controllers import blueprint
from controllers.middleware import auth, user
from flask import render_template

@blueprint.bp.route('/')
@auth
@user
def user_dashboard():
    return render_template('user/index.html')
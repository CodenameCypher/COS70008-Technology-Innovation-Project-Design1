from controllers import blueprint
from controllers.middleware import guest
from flask import render_template

@blueprint.bp.route('/')
@guest
def general_dashboard():
    return render_template('dashboard.html')
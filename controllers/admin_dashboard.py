from controllers import blueprint
from controllers.middleware import admin, auth
from flask import render_template

@blueprint.bp.route('/admin')
@auth
@admin
def admin_dashboard():
    return render_template('admin/index.html')
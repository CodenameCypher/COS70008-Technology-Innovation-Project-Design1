from controllers import blueprint
from controllers.middleware import admin, auth
from flask import flash, redirect
from models.user import User
from models.database import database

@blueprint.bp.route('/admin/manage/delete/<int:id>')
@auth
@admin
def delete_user(id):
    try:
        userObject = User.query.filter_by(id=id).delete()
        database.session.commit()

        flash("User deleted successfully!",'success')
    except:
        flash("Could not delete user!",'danger')
    return redirect('/admin/manage')
    
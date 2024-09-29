from controllers import blueprint
from controllers.middleware import auth
from flask import session, redirect, flash

@blueprint.bp.route('/logout')
@auth
def logout():
    session.pop("user_id",None)
    flash("Logged out successfully!",'success')
    return redirect('/login')
from controllers import blueprint
from controllers.middleware import auth, admin
from flask import render_template, request, redirect, session, flash
from models.user import User
import bcrypt
from models.database import database

@blueprint.bp.route('/admin/profile', methods=['GET', 'POST'])
@auth
@admin
def profile():
    userObject = User.query.filter_by(id=session['user_id']).first()
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        if name != '':
            userObject.name = name

        if email != '':
            userObject.email = email
        
        if password != '':
            userObject.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        try:
            database.session.commit()
            flash('Profile successfully updated.','success')
        except:
            flash('Profile update failed!','danger')

        return redirect('/admin/profile')
    else:
        return render_template('admin/profile.html', user=userObject)
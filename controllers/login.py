from controllers import blueprint
from controllers.middleware import guest
from flask import render_template, request, session, redirect, flash
from models.user import User

@blueprint.bp.route('/login', methods=['GET','POST'])
@guest
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        userObject = User.query.filter_by(email=email).first()
    
        if userObject and userObject.check_password(password):
            session['user_id'] = userObject.id
            flash("Welcome "+userObject.name+"!",'success')
            if userObject.isAdmin:
                return redirect('admin/')
            else: return redirect('/')
        else:
            flash("Authentication failed!",'danger')
            return redirect('/login')

    return render_template('authentication/login.html')
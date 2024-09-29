from controllers import blueprint
from controllers.middleware import guest
from flask import render_template, request, redirect, flash
from models.user import User

@blueprint.bp.route('/registration', methods=['GET', 'POST'])
@guest
def registration():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        new_user = User(name, email, password)

        try:
            new_user.save()
            flash("Registration successful!",'success')
        except:
            flash("Registration failed!",'danger')
        return redirect('/login')
    else:
        return render_template('authentication/register.html')

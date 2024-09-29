import functools
from flask import session, redirect, flash
from models.user import User

def auth(controller_function):
    @functools.wraps(controller_function)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('You need to be logged in to access this page!', 'danger')
            return redirect('/login')
        else:
            return controller_function(*args, **kwargs)
    return decorated

def guest(controller_function):
    @functools.wraps(controller_function)
    def decorated(*args, **kwargs):
        if 'user_id' in session:
            flash('You are already logged in!', 'success')
            user = User.query.filter_by(id=session['user_id']).first()
            if user.isAdmin:
                return redirect('/admin')
            else:
                return redirect('/')
        else:
            return controller_function(*args, **kwargs)
    return decorated


def admin(controller_function):
    @functools.wraps(controller_function)
    def decorated(*args, **kwargs):
        user = User.query.filter_by(id=session['user_id']).first()
        if not user.isAdmin:
            flash('You need to be an admin to access this page!', 'danger')
            return redirect('/')
        else:
            return controller_function(*args, **kwargs)
    return decorated

def user(controller_function):
    @functools.wraps(controller_function)
    def decorated(*args, **kwargs):
        user = User.query.filter_by(id=session['user_id']).first()
        if user.isAdmin:
            flash('You cannot access this page as an admin!', 'danger')
            return redirect('/admin')
        else:
            return controller_function(*args, **kwargs)
    return decorated
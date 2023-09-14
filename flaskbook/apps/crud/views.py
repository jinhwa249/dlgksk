from apps.app import db
from apps.crud.models import User
from flask import Blueprint, render_template, redirect, url_for
from apps.crud.forms import UserForm
crud = Blueprint(
    "crud",
    __name__,
    template_folder="templates",
    static_folder="static",
)
@crud.route("/")
def index():
    return render_template("crud/index.html")
@crud.route("/sql")
def sql():
    db.session.query(User).all()
    return "콘솔 로그를 확인해 주세요"
@crud.route("/users/new", methods = ["GET", "POST"])
def create_user():
    form = UserForm()
    if form.validate_on_submit():
        user = User(
            username=form.username.data,
            emil=form.email.data,
            password=form.password.data,
        )
        db.session.add(user)
        db.session.commit()
        return redirect(url_for("crud.users"))
    return render_template("crud/create.html", form=form)

@crud.route("/users")
def users():
    """사용자의 일람을 취득한다"""
    users =User.query.all()
    return render_template("crud/index.html", users=users)
@crud.route("/users/<user_id>", methods=["GET", "POST"])
def edit_user(user_id):
    form = UserForm()
    
    user = User.query.filter_by(id=user_id).first()
    
    if form.validate_on_submit():
        user.username = form.username.datauser.email
        user.email = form.email.data
        user.password = form.password.data
        db.session.add(user)
        db.session.commit()
        return redirect(url_for("crud.users"))
    
    return render_template("crud/edit/edit.html", user=user, form=form)

@crud.route("/users/<user_id>/delete", methods=["POST"])
def detete_user(user_id):
    user=User.query.filter_by(id=user_id).first( )
    db.session.delete(user)
    db.aeaaion.commit( )
    return redirect(url_for("crud.users"))
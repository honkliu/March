from flask_wtf import FlaskForm
from wtforms import TextField, BooleanField, PasswordField, SubmitField
from wtforms.validators import DataRequired

class LoginForm(FlaskForm) :
    username = TextField('Name', validators=[DataRequired()])
    password = PasswordField('Pass', validators=[DataRequired])
    remember_me = BooleanField('remember_me', default=False)
    submit = SubmitField('Tijiao') 

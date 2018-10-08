from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os

app = Flask(__name__)

basedir = os.path.abspath(os.path.dirname(__file__))
app.config['CSRF_ENABLED'] = True
app.config['SECRET_KEY'] = 'you guess'
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL') or 'sqlite:///' + os.path.join(basedir, 'app.db')

db = SQLAlchemy(app)

migrate = Migrate(app, db)

from app import views, usermodel



from flask import render_template
from app import app
from .forms import LoginForm

@app.route('/')
@app.route('/index')
def index() :
	user = {'name' : 'Shuguang Liu' }
	posts = [
		{
			'author': {'nickname' : 'Yun Ma'},
			'body' : 'I am mom'
		},
		{
			'author' : {'nickname' : "Yiru Liu"},
			'body' : 'I am daughter'
		}
	]
	return render_template("index.html",
		title='Board',
		user = user,
		posts = posts) 

@app.route('/login', methods = ['GET', 'POST'])
def login() :
	form = LoginForm()

	return render_template("login.html",
			title = 'Sign in',
			form = form)

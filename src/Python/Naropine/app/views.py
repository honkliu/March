from flask import render_template
from app import app
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
	

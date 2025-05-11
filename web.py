from flask import Flask, url_for, request
import requests

app = Flask(__name__)

@app.route('/')
def index():
  return "Index page"

@app.route('/ai', methods=["POST"])
def ai():
  return str(request.json['number'])



if __name__ == '__main__':
  app.run(debug=True)

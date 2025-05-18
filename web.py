from flask import Flask, url_for, request, render_template
import process
import requests

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/ai', methods=["POST"])
def ai():
  pixels = request.json["pixels"]
  print(process.image_str(pixels))
  response = process.recognise(pixels)
  maxNo, maxProb = 0, 0
  for i in range(len(response)):
    if int(response[i]) > maxProb:
      maxNo = i
      maxProb = response[i]
    
  return {"response": response, "maxNo": maxNo}


if __name__ == '__main__':
  app.run(debug=True)

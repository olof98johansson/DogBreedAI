import os

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

import numpy as np
import pandas as pd
import argparse
import imutils
import cv2
import time
import uuid
import base64

import torch
import torch.nn as nn
import torchvision.models
import requests
import cv2
import setup
from torchvision import transforms



num_classes = setup.config.num_classes
size = setup.config.size
model = setup.get_model(num_classes)


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)

def get_breed(label):
  breeds = pd.read_csv(setup.config.sample_sub_path).keys()[1:]
  labels = range(len(breeds))
  breed_to_label = dict(zip(breeds, labels))
  label_to_breed = dict(zip(labels, breeds))

  for item in label_to_breed.items():
    if item[0] == label:
      return item[1]
  return None



def predict(file):
  img = cv2.imread(file)
  #img = cv2.resize(img, (size,size))
  #img = np.rollaxis(img, 2, 0)
  #img = np.expand_dims(img, axis=0)
  #img = torch.Tensor(img)
  #mean, std = img.mean(), img.std()
  #normal = transforms.Normalize(mean.unsqueeze(0), std.unsqueeze(0))
  #normal(tensor=img)
  #img = np.array(img)
  transf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
  img = transf(img)
  img = img.unsqueeze(0)
  out = model(img)
  _, predicted = torch.max(out.data, 1)
  predicted = predicted.numpy()
  breed = get_breed(predicted)
  print(f"Label: {str(breed)}")

  return predicted



def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
    return render_template('template.html', label='', imagesource='https://storage.googleapis.com/burbcommunity-morethanthecurve/2014/03/pluto-dog.jpg')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        import time
        start_time = time.time()
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            result = predict(file_path)
            label = str(get_breed(result))

            print(result)
            print(file_path)
            filename = my_random_string(6) + filename

            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("--- %s seconds ---" % str (time.time() - start_time))
            return render_template('template.html', label=label, imagesource='../uploads/' + filename)
        else:
            #return render_template('template.html', label='', imagesource= 'https://storage.googleapis.com/burbcommunity-morethanthecurve/2014/03/pluto-dog.jpg')
            return None
from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
from werkzeug.middleware.shared_data import SharedDataMiddleware
#from werkzeug import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})

if __name__ == "__main__":
    app.debug=True
    app.run(host='0.0.0.0', port=3000)

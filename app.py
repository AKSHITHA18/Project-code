import os
from PIL import Image
import numpy as np      # Importing the libraries
import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVC
from sklearn import metrics
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request
import cv2
import numpy
from easyocr import Reader

import enchant
e= enchant.Dict("en_US")
import re
import pyttsx3
from flask_cors import cross_origin
from flask import Flask, render_template, request
from main import text_to_speech

def text_to_speech(text, gender):
    """
    Function to convert text to speech
    :param text: text
    :param gender: gender
    :return: None
    """
    voice_dict = {'Male': 0, 'Female': 1}
    code = voice_dict[gender]

    engine = pyttsx3.init()

    # Setting up voice rate
    engine.setProperty('rate', 125)

    # Setting up volume level  between 0 and 1
    engine.setProperty('volume', 0.8)

    # Change voices: 0 for male and 1 for female
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[code].id)

    engine.say(text)
    engine.runAndWait()
    
r = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
l=['en']
# print("[INFO] OCR'ing with the following languages: {}".format(l))

reader = Reader(l, gpu=False)


app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
def fun(image):
    results = reader.readtext(image)
    vw=[];vph=[]
    for i in results:
        if i[1]!="":
            if e.check(i[1]):
                vw.append(i[1])
            else:
                s=r.findall(i[1])
                vph.append(s)
                l=list(map(str,i[1].split()))
                for j in l:
                    if e.check(j):
                        vw.append(j)
    vph=[]
    for i in results:
        s=r.findall(i[1])
        vph.append(s)
    fin_ph=[]
    for i in vph:
        for j in i:
            fin_ph.append(j)
    ph_num=[];vww=[]
    for i in vw:
        if i[0]=='9' or i[0]=='8' or i[0]=='7' or i[0]=='6':
            if len(i)==10 or len(i)==12:
                ph_num.append(i)
        elif i[0].isnumeric()!=True:
            vww.append(i)
    for i in fin_ph:
        if len(i)==7:
            ph_num.append(i)
        elif i[0]=='9' or i[0]=='8' or i[0]=='7' or i[0]=='6' :
            ph_num.append(i)
    return ph_num,vw
        
        
def mirror_this(image_file, gray_scale=False, with_plot=False):
    image_rgb = read_this(image_file=image_file, gray_scale=gray_scale)
    image_mirror = np.fliplr(image_rgb)

    return image_mirror    
from matplotlib import pyplot as plt
def read_this(image_file, gray_scale=False):
    image_src = cv2.imread(image_file)
    if gray_scale:
        image_rgb = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
    else:
        image_rgb = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
    return image_rgb
  
        
        
@app.route('/')
def index():
    return render_template("check.html")

@app.route('/upload', methods=['POST'])
def upload():
    return render_template("upload.html")

@app.route('/predict', methods=['POST'])

def predict():
    
    for file in request.files.getlist("file"):
        print(file)
        
        #read image file string data
        filestr = file.read()
        #convert string data to numpy array
        npimg = numpy.fromstring(filestr, numpy.uint8)
        # convert numpy array to image
        originalImage = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)

        flipV = cv2.flip(originalImage, 0)
        flipH = cv2.flip(originalImage, 1)
        flipB = cv2.flip(originalImage, -1)
        res1,vw1=fun(originalImage)
        res2,vw2=fun(originalImage)
        res3,vw3=fun(originalImage)
        res4,vw4=fun(originalImage)
        res5,vw5=fun(originalImage)
        
        l=res1+res2+res3+res4+res5
        l=set(l)
        vw=vw1+vw2+vw3+vw4+vw5
        vw=set(vw)
        fvw=[]
        for i in vw:
            if i[0]!='0' and  i[0]!='1' and i[0]!='2' and i[0]!='3' and i[0]!='4' and i[0]!='5' and i[0]!='6' and i[0]!='7' and i[0]!='8' and i[0]!='9':
                fvw.append(i)
        if(fvw==[]):
            fvw="No Words Found"
        if l==[]:
            l="No Phone Numbers Found"
        
    return render_template("prediction.html",ph='predicted Phone Numbers: {}'.format('\n,'.join(l)),f_vw='predicted text:  "{}"'.format('\n,'.join(fvw)))

@app.route('/voice', methods=['POST'])
def voice():
    return render_template("frontend.html")

@app.route('/speak', methods=['POST','GET'])
   
def speak():
    text = request.form['speech']
    gender = request.form['voices']
    text_to_speech(text, gender)
    return render_template('frontend.html')
if __name__ == "__main__":
    app.run(port=4555, debug=True)
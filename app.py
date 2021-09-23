from flask import Flask, render_template, request, session, url_for, redirect, jsonify
import pymysql
import numpy as np
#import pandas as pd
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

emotionlistforadding=['Angry','Disgust','Fearful','Happy','Neutral','Sad','Surprise']
emoji_dist={0:"./emojis/angry.png",2:"./emojis/disgusted.png",2:"./emojis/fearful.png",3:"./emojis/happy.png",4:"./emojis/neutral.png",5:"./emojis/sad.png",6:"./emojis/surpriced.png"}
global last_frame1                                    
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text=[0]
global listofemotion
listofclasses=['ENFJ',
 'ENFP',
 'ENTJ',
 'ENTP',
 'ESFJ',
 'ESFP',
 'ESTP',
 'INFJ',
 'INFP',
 'INTJ',
 'INTP',
 'ISFJ',
 'ISFP',
 'ISTJ',
 'ISTP']
import os
import cv2
import numpy as np
import tensorflow as tf
from model import FacialExpressionModel

# Creating an instance of the class with the parameters as model and its weights.
test_model = FacialExpressionModel("model.json", "model_weights.h5")

# Loading the classifier from the file.
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
tf.config.experimental.list_physical_devices('GPU')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB * 2 of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 2)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
cv2.ocl.setUseOpenCL(False)
app = Flask(__name__)
app.secret_key = 'random string'


def dbConnection():
    try:
        connection = pymysql.connect(host="localhost", user="root", password="root", database="029cognitive-emotion")
        return connection
    except:
        print("Something went wrong in database Connection")


def dbClose():
    try:
        dbConnection().close()
    except:
        print("Something went wrong in Close DB Connection")

model = model_from_json(open("fer.json", "r").read())
#load weights
model.load_weights('fer.h5')
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def startcamera():
    global listofemotion
    listofemotion=[]
    cap = cv2.VideoCapture(0)
    while True:
        Text=None
        ret,image=cap.read()# captures frame and returns boolean value and captured image
        #print('hello')
        if not ret:
            continue
        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Image size is reduced by 30% at each image scale.
        scaleFactor = 1.3

    # 5 neighbors should be present for each rectangle to be retained.
        minNeighbors = 5

    # Detect the Faces in the given Image and store it in faces.
        faces = facec.detectMultiScale(gray_frame, scaleFactor, minNeighbors)

    # When Classifier could not detect any Face.


        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Taking the Face part in the Image as Region of Interest.
            roi = gray_frame[y:y+h, x:x+w]

        # Let us resize the Image accordingly to use pretrained model.
            roi = cv2.resize(roi, (48, 48))

        # Let us make the Prediction of Emotion present in the Image
            prediction = test_model.predict_emotion(
                roi[np.newaxis, :, :, np.newaxis])

        # Custom Symbols to print with text of emotion.
            Symbols = {"Happy": ":)", "Sad": ":}", "Surprise": "!!",
                       "Angry": "?", "Disgust": "#", "Neutral": ".", "Fear": "~"}
    

        ## based on the prediction recommend music


        # Defining the Parameters for putting Text on Image
            Text = str(prediction)
            Text_Color = (180, 105, 255)

            Thickness = 2
            Font_Scale = 1
            Font_Type = cv2.FONT_HERSHEY_SIMPLEX

        # Inserting the Text on Image
            cv2.putText(image, Text, (x, y), Font_Type,
                    Font_Scale, Text_Color, Thickness)
            listofemotion.append(Text)
            imagetoread=cv2.imread('emojis//'+Text+".png")
            #cv2.imshow('Emoji found ',imagetoread)
        resized_img = cv2.resize(image, (1000, 700))
        cv2.imshow('Facial emotion analysis ',resized_img)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()
    
@app.route('/index')
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/logout')
def logout():
    session.pop('user')
    return redirect(url_for('index'))
@app.route('/clickvideoemotion', methods = ['GET', 'POST'])
def clickvideoemotion():
    if 'user' in session:
        if request.method == 'POST':
            global listofemotion
            link = request.form.get("link")
            print('link is',link)
            #fbposts = FB_post_fetch(fblink) 
            #fbposts=LinkedInfetchingscrapinguserdata(link)
            fbposts=[]
            startcamera()
            countallemotion={}
            username=session['user']
            con = dbConnection()
            cursor = con.cursor()
            
            for ik in emotionlistforadding:
                valis=listofemotion.count(ik)
                
                
                countallemotion[ik]=valis
                sql = "INSERT INTO emotion ( username, emotion,count) VALUES (%s, %s, %s)"
                val = (username, ik,  valis)
                cursor.execute(sql, val)
                con.commit()
                print(countallemotion)
                print(listofemotion)
            return render_template('Fetchedemotion.html', data=countallemotion, user=session['user'])
        return render_template('home.html', user=session['user'])
    return redirect(url_for('index'))

@app.route('/login', methods=["GET","POST"])
def login():
    msg = ''
    if request.method == "POST":
        try:
            session.pop('user',None)
            username = request.form.get("email")
            password = request.form.get("pass")
            con = dbConnection()
            cursor = con.cursor()
            cursor.execute('SELECT * FROM userdetails WHERE email = %s AND password = %s', (username, password))
            result = cursor.fetchone()
            if result:
                session['user'] = result[1]
                session['userid'] = result[0]
                return redirect(url_for('home'))
            else:
                return redirect(url_for('index'))
        except:
            print("Exception occured at login")
            return redirect(url_for('index'))
        finally:
            dbClose()
    #return redirect(url_for('index'))
    return render_template('login.html',)


@app.route('/register', methods=["GET","POST"])
def register():
    if request.method == "POST":
        try:
            name = request.form.get("name")
            email = request.form.get("email")
            mobile = request.form.get("mobile")
            password = request.form.get("pass")
            con = dbConnection()
            cursor = con.cursor()
            sql = "INSERT INTO userdetails (name, email, mobile, password) VALUES (%s, %s, %s, %s)"
            val = (name, email, mobile, password)
            cursor.execute(sql, val)
            con.commit()
            return redirect(url_for('index'))
        except:
            print("Exception occured at login")
            return render_template('register.html')
        finally:
            dbClose()
    return render_template('register.html')

@app.route('/admin',methods=["GET","POST"])
def admin():
    return render_template('admin.html')


@app.route('/adminlogin', methods=["GET","POST"])
def adminlogin():
    if request.method == "POST":
        try:
            #session.pop('user',None)
            username = request.form.get("username")
            #print(username)
            password = request.form.get("password")
            #print(password)
            con = dbConnection()
            cursor = con.cursor()
            cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (username, password))
            result = cursor.fetchone()
            if result:
                session['admin_name'] = result[1]
                session['adminid'] = result[0]
                #flash("Successfully login")
                return redirect(url_for('adminhome'))
            else:
                #flash("(Invalid mail id and password)")
                return redirect(url_for('admin'))
        except Exception as e:
            print(e)
            print("Exception occured at login")
            #flash("Something went wrong")
            return redirect(url_for('admin'))
    return render_template('admin')


@app.route('/adminhome',methods=["GET","POST"])
def adminhome():
    if 'adminid' in session:
        admin_name = session['admin_name']
        return render_template('adminhome.html', uname=admin_name)
    return redirect(url_for('admin'))


@app.route('/stats',methods=["GET","POST"])
def stats():
    if 'adminid' in session:
        admin_name = session['admin_name']
        con = dbConnection()
        cursor = con.cursor()
        cursor.execute('SELECT * FROM userdetails')
        result = cursor.fetchall()
        return render_template('stats.html', result=result, uname=admin_name)
    return redirect(url_for('admin'))

from collections import Counter
@app.route('/userstats',methods=["GET","POST"])
def userstats():
    if 'adminid' in session:
        admin_name = session['admin_name']
        con = dbConnection()
        cursor = con.cursor()
        cursor.execute('SELECT * FROM emotion')
        result = list(cursor.fetchall())
        list1=[]
        counts={}
        for ik in result:
            list1.append(ik[0])
            counts[ik[2]]=int(ik[3])
            
        #list1 =result[0]# ['sad', 'happy', 'depressed', 'sad', 'depressed','sad', 'happy', 'depressed']
        #counts = dict(Counter(list1))
        print(counts)

        return render_template('userstats.html', counts=counts, uname=admin_name)
    return redirect(url_for('admin'))

@app.route('/home')
def home():
    if 'user' in session:
        return render_template('home.html', user=session['user'])
    return redirect(url_for('index'))


#if __name__ == '__main__':
    #app.run('0.0.0.0')
    #app.run()
startcamera()
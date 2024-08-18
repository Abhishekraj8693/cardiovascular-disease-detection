import numpy as np  # dealing with arrays
import os  # dealing with directories
import tensorflow
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.api.models import Sequential, load_model
from keras.api.optimizers import Adam
from flask import Flask, render_template, request
import sqlite3
import cv2
import shutil

from cnn_vita import model

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diat')
def diat():
    return render_template('diat.html')

@app.route('/diat1')
def diat1():
    return render_template('diat1.html')
@app.route('/logout')
def logout():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('home.html')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']

        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')

    return render_template('index.html')

@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':


        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"


        shutil.copy("test\\"+fileName, dst)

        verify_dir = 'static/images'
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'VITAMIN-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
        def process_verify_data():
            verifying_data = []
            for img in os.listdir(verify_dir):
                path = os.path.join(verify_dir, img)
                img_num = img.split('.')[0]
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                verifying_data.append([np.array(img), img_num])
                np.save('verify_data.npy', verifying_data)
            return verifying_data

        def build_model():
            model = Sequential()

            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
            model.add(MaxPooling2D(pool_size=(3, 3)))

            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(3, 3)))

            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(3, 3)))

            model.add(Conv2D(32, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(3, 3)))

            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(3, 3)))

            model.add(Flatten())
            model.add(Dense(1024, activation='relu'))
            model.add(Dropout(0.8))

            model.add(Dense(2, activation='softmax'))  # 2 classes for binary classification

            optimizer = Adam(learning_rate=LR)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

            return model

        if os.path.exists(MODEL_NAME):
            model = load_model(MODEL_NAME)
            print('Model loaded!')
        else:
            model = build_model()

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        accuracy=" "
        str_label=" "
        verify_data = process_verify_data()
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            #y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            if np.argmax(model_out) == 0:
                str_label = 'VITAMIN DEFICIENCY B'
                print("The predicted image of the skin has vitamin deficiency B with a accuracy of {} %".format(model_out[0]*100))
                accuracy = "The predicted image of the skin has vitamin deficiency B  with a accuracy of {} %".format(model_out[0]*100)

            elif np.argmax(model_out) == 1:
                str_label = 'VITAMIN DEFICIENCY C'
                print("The predicted image of the skin has vitamin deficiency C with a accuracy of {} %".format(model_out[1]*100))
                accuracy = "The predicted image of the skin has vitamin deficiency C with a accuracy of {} %".format(model_out[1]*100)



        return render_template('result.html', status=str_label,accuracy=accuracy, ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName)
    return render_template('home.html')

if __name__ == "__main__":

    app.run(debug=True, use_reloader=False)

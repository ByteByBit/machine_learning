from threading import Thread
from queue import Queue, Empty

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

import numpy as np

from PIL import ImageGrab


class ImageUtil:

    ''' Class to have static image utility methods. '''

    @staticmethod
    def get_img(drawing_area):
        ''' Get image from UI and prepare (resize, grayscale) '''

        x, x1, y, y1 = drawing_area
        img = ImageGrab.grab().crop((x,y,x1,y1)).convert("L")

        img = img.resize((28,28))
        img.convert('LA')

        return img

    @staticmethod
    def prepare_image(img):
        ''' Prepare image for Tensorflow. '''

        img_arr = keras.preprocessing.image.img_to_array(img)
        img = np.array([img_arr])
        img = img.reshape((img.shape[0], 28, 28, 1)).astype('float32')

        return img


class TrainModel(Thread):
        
    ''' Train Tensorflow's model. '''

    def __init__(self, work_q, df):

        Thread.__init__(self)
        self.q = work_q
        self.DATA_FILE = df

    def load_data(self):
        ''' Load MNIST data for training. '''

        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

    def prepare_date(self):
        ''' Preprocess data to fit to Tensorflow. '''

        # reshape to be [samples][width][height][channels]
        self.x_train = self.x_train.reshape((self.x_train.shape[0], 28, 28, 1)).astype('float32')
        self.x_test = self.x_test.reshape((self.x_test.shape[0], 28, 28, 1)).astype('float32')

        # normalize inputs from 0-255 to 0-1
        self.x_train = self.x_train / 255
        self.x_test = self.x_test / 255

        # one hot encode outputs
        self.y_train = np_utils.to_categorical(self.y_train)
        self.y_test = np_utils.to_categorical(self.y_test)
        self.num_classes = self.y_test.shape[1]

        self.create_model()

    def create_model(self):
        ''' Init and set model. '''

        self.model = Sequential()
        self.model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
        self.model.add(MaxPooling2D())
        self.model.add(Conv2D(15, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D())
        self.model.add(Dropout(0.2))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        # Compile model
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.fit_model()

    def fit_model(self):
        ''' Fit model to data. '''

        self.model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=10, batch_size=200)

        self.eval_model()
    
    def eval_model(self):
        ''' Evaluate model. '''

        self.scores = self.model.evaluate(self.x_test, self.y_test, verbose=0)

    def save_model(self):
        ''' Save trained model. '''

        self.model.save(self.DATA_FILE)
        # Put the result into a Q in order to use it in the main thread.
        self.q.put(self.scores)


class DigitRecognizer():
    
    ''' Class to handle ML. '''

    def __init__(self):

        self.progress = False
        self.q = Queue()

    def process(self):
        ''' Process image. '''

        da = self.get_drawing_area()
        img = ImageUtil.get_img(drawing_area=da)

        prediction = self.start_prediction(img)

        # Display result.
        self.display_msg(title='Prediction', text=prediction)
        self.drawing_area.delete('all')

    def start_prediction(self, img):
        ''' Predict the digit. '''

        image = ImageUtil.prepare_image(img)
        self.model = keras.models.load_model(self.c.DATA_FILE)
        predict = self.model.predict(image)
        res = str(np.argmax(predict))
        return res

    
    def start_training(self):
        ''' Start training the model. '''

        tm = TrainModel(work_q=self.q, df=self.c.DATA_FILE) 
        t = Thread(target=tm.load_data)
        t.start()

    def process_q(self):
        ''' Process the Q to get Accuracy and Loss back. '''

        try:

            res = self.q.get(0) # (Loss, Accuracy)

            txt = 'Loss: {}\n, Accuracy: {}'.format(res[0], res[1])
            self.display_msg(title='Loss & Accuracy', text=txt)
            self.progress = True

        except Empty: # Thread is not ready yet.
            self.parent.after(1000, self.process_q)
from keras.callbacks import LearningRateScheduler,EarlyStopping, ModelCheckpoint
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='nb_epoch', type=int, default=5, help='# of epochs')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=100, help='# images in batch')
parser.add_argument('--verbose', dest='verbose', type=int, default=1, help='verbose type')

#args = parser.parse_args()
args = vars(parser.parse_args())
np.random.seed(222)

nb_classes = 10

img_rows, img_cols = 28, 28

nb_filters = 32

pool_size = (2, 2)

kernel_size = (3, 3)


(X_train, y_train), (X_test, y_test) = mnist.load_data()

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

def main():

    sd=[]
    class LossHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = [1,1]
    
        def on_epoch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
            sd.append(step_decay(len(self.losses)))
    
    
    def step_decay(losses):
        if float(2*np.sqrt(np.array(history.losses[-1])))<1.6:
            lrate=0.001
            return lrate
        else:
            lrate=0.01
            return lrate

    model = Sequential()
    
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
    		                border_mode='valid',
    		                input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',optimizer='adadelta',
                  metrics=['accuracy'])
    history=LossHistory()
    lrate=LearningRateScheduler(step_decay)
    filepath="_project-weights-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True)
    
    
    model.fit(X_train, Y_train,args["nb_epoch"],args["batch_size"],callbacks=[history,lrate,checkpoint],verbose=args["verbose"])
    
#    filename = "_project-weights-01-2.3003.hdf5"
#    model.load_weights(filename)
#    model.compile(loss='mean_squared_error', optimizer='adam')
    return print("Accuracy:",model.evaluate(X_test, Y_test, verbose=1))

if __name__ == '__main__':
   main()

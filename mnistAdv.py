import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from matplotlib import pyplot as plt
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_tf import model_eval
import tensorflow as tf

sess = tf.Session()
keras.backend.set_session(sess)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#tensorflow expects explicit channel dimension so we add it here (x1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

#as floats
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#between 0 and 1
x_train /= 255
x_test /= 255

#categorical classes (vectors of 1x10, one-hot)
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=1)
score = model.evaluate(x_test, y_test, verbose=0)

wrap = KerasModelWrapper(model)
fgsm = FastGradientMethod(wrap, sess=sess)
fgsm_params = {'eps': 0.3,
               'clip_min': 0.,
               'clip_max': 1.}
x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
y = tf.placeholder(tf.float32, shape=(None, 10))
adv_x = fgsm.generate(x, **fgsm_params)
adv_x = tf.stop_gradient(adv_x) #Consider the attack to be constant
preds_adv = model(adv_x)
eval_par = {'batch_size': 32}
acc = model_eval(sess, x, y, preds_adv, x_test, y_test, args=eval_par)
print('Test accuracy on adversarial examples: %0.4f\n' % acc)


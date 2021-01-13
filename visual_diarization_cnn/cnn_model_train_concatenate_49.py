from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

import numpy as np
from keras import backend as K

K.image_data_format()


if K.image_data_format() == 'channels_first':
    input_shape = (3, 224, 224)
else:
    input_shape = (224, 224, 3)


dirname = ['NET20070412_thlep_1_1', 'NET20070330_thlep_1_2', 'NET20070401_thlep_1_1', 'NET20070330_thlep_1_3', 'NET20070331_thlep_1_1', 'NET20070412_thlep_1_2', 'NET20070402_thlep_1_1', 'NET20070402_thlep_1_2', 'NET20070403_thlep_1_1', 'NET20070330_thlep_1_4', 'NET20070326_thlep_1_1', 'NET20070329_thlep_1_1', 'NET20070330_thlep_1_1', 'NET20070331_thlep_1_2']

x_train_0 = loaded_array = np.load('samples&targets_concatenate_49/' + dirname[0] + '/x_concatenate.npy')
x_train_1 = loaded_array = np.load('samples&targets_concatenate_49/' + dirname[1] + '/x_concatenate.npy')
x_train_2 = loaded_array = np.load('samples&targets_concatenate_49/' + dirname[2] + '/x_concatenate.npy')
x_train_3 = loaded_array = np.load('samples&targets_concatenate_49/' + dirname[3] + '/x_concatenate.npy')
x_train_4 = loaded_array = np.load('samples&targets_concatenate_49/' + dirname[4] + '/x_concatenate.npy')
x_train_5 = loaded_array = np.load('samples&targets_concatenate_49/' + dirname[5] + '/x_concatenate.npy')
x_train_6 = loaded_array = np.load('samples&targets_concatenate_49/' + dirname[6] + '/x_concatenate.npy')
x_train_7 = loaded_array = np.load('samples&targets_concatenate_49/' + dirname[7] + '/x_concatenate.npy')
x_train_8 = loaded_array = np.load('samples&targets_concatenate_49/' + dirname[8] + '/x_concatenate.npy')

x_test_0 = loaded_array = np.load('samples&targets_concatenate_49/' + dirname[9] + '/x_concatenate.npy')
# x_test_1 = loaded_array = np.load('samples&targets_concatenate_49/' + dirname[10] + '/x_concatenate.npy')
# x_test_2 = loaded_array = np.load('samples&targets_concatenate_49/' + dirname[11] + '/x_concatenate.npy')
# x_test_3 = loaded_array = np.load('samples&targets_concatenate_49/' + dirname[12] + '/x_concatenate.npy')
# x_test_4 = loaded_array = np.load('samples&targets_concatenate_49/' + dirname[13] + '/x_concatenate.npy')

x_train_0 = x_train_0 / 255
x_train_1 = x_train_1 / 255
x_train_2 = x_train_2 / 255
x_train_3 = x_train_3 / 255
x_train_4 = x_train_4 / 255
x_train_5 = x_train_5 / 255
x_train_6 = x_train_6 / 255
x_train_7 = x_train_7 / 255
x_train_8 = x_train_8 / 255

x_test_0 = x_test_0 / 255
# x_test_1 = x_test_1 / 255
# x_test_2 = x_test_2 / 255
# x_test_3 = x_test_3 / 255
# x_test_4 = x_test_4 / 255

y_train_0 = loaded_array = np.load('samples&targets_concatenate_49/' + dirname[0] + '/y_concatenate.npy')
y_train_1 = loaded_array = np.load('samples&targets_concatenate_49/' + dirname[1] + '/y_concatenate.npy')
y_train_2 = loaded_array = np.load('samples&targets_concatenate_49/' + dirname[2] + '/y_concatenate.npy')
y_train_3 = loaded_array = np.load('samples&targets_concatenate_49/' + dirname[3] + '/y_concatenate.npy')
y_train_4 = loaded_array = np.load('samples&targets_concatenate_49/' + dirname[4] + '/y_concatenate.npy')
y_train_5 = loaded_array = np.load('samples&targets_concatenate_49/' + dirname[5] + '/y_concatenate.npy')
y_train_6 = loaded_array = np.load('samples&targets_concatenate_49/' + dirname[6] + '/y_concatenate.npy')
y_train_7 = loaded_array = np.load('samples&targets_concatenate_49/' + dirname[7] + '/y_concatenate.npy')
y_train_8 = loaded_array = np.load('samples&targets_concatenate_49/' + dirname[8] + '/y_concatenate.npy')
'''
y_train_0 = y_train_0[0:int(y_train_0.shape[0]/2)]
y_train_1 = y_train_1[0:int(y_train_1.shape[0]/2)]
y_train_2 = y_train_2[0:int(y_train_2.shape[0]/2)]
y_train_3 = y_train_3[0:int(y_train_3.shape[0]/2)]
y_train_4 = y_train_4[0:int(y_train_4.shape[0]/2)]
y_train_5 = y_train_5[0:int(y_train_5.shape[0]/2)]
y_train_6 = y_train_6[0:int(y_train_6.shape[0]/15)]
'''
y_test_0 = loaded_array = np.load('samples&targets_concatenate_49/' + dirname[9] + '/y_concatenate.npy')
# y_test_1 = loaded_array = np.load('samples&targets_concatenate_49/' + dirname[10] + '/y_concatenate.npy')
# y_test_2 = loaded_array = np.load('samples&targets_concatenate_49/' + dirname[11] + '/y_concatenate.npy')
# y_test_3 = loaded_array = np.load('samples&targets_concatenate_49/' + dirname[12] + '/y_concatenate.npy')
# y_test_4 = loaded_array = np.load('samples&targets_concatenate_49/' + dirname[13] + '/y_concatenate.npy')


x_train = np.concatenate((x_train_0, x_train_1, x_train_2, x_train_3, x_train_4, x_train_5, x_train_6, x_train_7, x_train_8), axis=0)
y_train = np.concatenate((y_train_0, y_train_1, y_train_2, y_train_3, y_train_4, y_train_5, y_train_6, y_train_7, y_train_8), axis=0)

x_test = x_test_0
y_test = y_test_0

x_test = x_test.reshape(x_test.shape[0], 224, 224, 3)
y_test = y_test.reshape(x_test.shape[0], 1)

print(x_train.shape, y_train.shape)

x_train = x_train.reshape(x_train.shape[0], 224, 224, 3)
y_train = y_train.reshape(x_train.shape[0], 1)

print(x_train.shape)
print(y_train.shape)

model = Sequential()

model.add(Convolution2D(64, kernel_size=3, input_shape=input_shape, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.3))

model.add(Convolution2D(128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.3))

model.add(Convolution2D(256, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.3))

model.add(Convolution2D(256, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.3))

model.add(Convolution2D(256, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))

opt = Adam(lr=0.001, decay=1e-6)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

class_weight = {0: 5, 1: 1}
history = model.fit(x_train, y_train, class_weight=class_weight, validation_split=0.1, validation_data=(x_test, y_test), batch_size=256, epochs=70)

print(model.summary())

results = model.evaluate(x_test, y_test, batch_size=256)
print('test loss, test acc:', results)

# model.evaluate(x_test, y_test)
model.save('model/49_cnn_model.h5')

model.save_weights('model/49_cnn_model_weights.h5')

print('\nhistory dict:', history.history)
fig_acc = plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_acc.savefig('accuracy_49')
plt.close(fig_acc)

fig_loss = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('loss_49')
plt.close(fig_loss)

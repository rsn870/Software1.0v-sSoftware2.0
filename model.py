

import numpy as np



  


def fizz_vec(N) :
    if N % 15 == 0:
        return np.array([1, 0, 0, 0])
    elif N % 5 == 0:
        return np.array([0, 1, 0, 0])
    elif N % 3 == 0:
        return np.array([0, 0, 1, 0])
    else:
        return np.array([0, 0, 0, 1])



import sklearn

num_digits=16
def input_vec(N):
  return np.array([N >> d & 1 for d in range(num_digits)])

X = np.array([input_vec(i) for i in range(101, 240)])
x1 =np.array([input_vec(i) for i in range(255,1000,15)])
x2 =np.array([input_vec(i) for i in range (240,600,5)])
x3= np.array([input_vec(i) for i in range(243,400,3)])
X = np.concatenate((X,x1))
X = np.concatenate((X,x2))
X = np.concatenate((X,x3))

y = np.array([fizz_vec(i) for i in range(101, 240)])
y1 =np.array([fizz_vec(i) for i in range(255,1000,15)])
y2 =np.array([fizz_vec(i) for i in range (240,600,5)])
y3= np.array([fizz_vec(i) for i in range(243,400,3)])
y = np.concatenate((y,y1))
y = np.concatenate((y,y2))
y = np.concatenate((y,y3))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)

import tensorflow

from tensorflow import keras

from tensorflow.keras import layers

inputs = keras.Input(shape=(16,) , name ='numbers')
x =layers.Dense(250 , activation ='relu')(inputs)
#x =layers.Dense(20 , activation ='tanh')(x)
x =layers.Dense(250 , activation ='relu')(x)
#x =layers.Dense(180 , activation ='relu')(x)
#x =layers.Dense(220 , activation ='relu')(x)
#x =layers.Dense(180 , activation ='relu')(x)
#x = layers.Dense(4, activation ='elu')(x)
outputs = layers.Dense(4, activation ='softmax')(x)
trainer= keras.Model(inputs = inputs , outputs =outputs)

x_val=X_train[-100:]
y_val=y_train[-100:]
x_train=X_train[:-100]
y_tr=y_train[:-100]

trainer.compile(optimizer = keras.optimizers.Adam(),loss = keras.losses.categorical_crossentropy )

history =trainer.fit(x_train , y_tr ,  epochs =700, validation_data=(x_val,y_val))
evaluations =trainer.evaluate(X_test,y_test)

trainer.save('model.h5')

del(trainer)



import numpy as np
from keras import layers

from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.models import Model, Sequential


activation = ["linear", #raw input
              "relu", #max(x, 0)
              "sigmoid", # sigmoid(x) = 1 / (1 + exp(-x))
              "softmax", #exp(x) / sum(exp(x))
              "softplus", #log(exp(x) + 1)
              "softsign", #x / (abs(x) + 1)
              "tanh", #sinh(x) / cosh(x)
              "hard_sigmoid", #0 if x < -2.5  || 1 if x > 2.5 || 0.2 * x + 0.5 if -2.5 <= x <= 2.5
              ]


maxEpoch = 10
i = 0

#functional
#inputs = Input(shape=(2,))
#hidden1 = Dense(units=2)(inputs) # hidden layer 1
#hidden2 = Dense(units=2,activation=activation[i])(hidden1) # hidden layer 2
#outputs = Dense(units=1,activation=outputs[i])(hidden2)
#model = Model(inputs=inputs,outputs=outputs)


#sequential
model = Sequential()
model.add(Input(shape=(2,)))
model.add(layers.Dense(units=2, activation=activation[i]))
model.add(layers.Dense(units=1))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='nadam')


ll = len(model.layers)
lw = len(model.weights)
print("numbers of layers = ", ll)
print("Number of weights after calling the model:",lw)



#OR
inputs = np.asarray([[1,1], [0,0], [1,0], [0,1]])
expected = np.asarray([[1], [0], [1], [1]])
model.fit(
    inputs,
    expected, 
    epochs=maxEpoch
)




#inputs = tf.keras.Input(shape=(3,))
#x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
#outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
#model = tf.keras.Model(inputs=inputs, outputs=outputs)
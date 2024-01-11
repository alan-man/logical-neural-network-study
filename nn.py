import numpy as np
from keras import layers

from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import SGD



from prettytable import PrettyTable
table = PrettyTable()
table.field_names = ["Num Hidden Layers","Hidden Layer Size","ePoch","Activation Function","Accuracy"]

activation = ["linear", #raw input
              "relu", #max(x, 0)
              "sigmoid", # sigmoid(x) = 1 / (1 + exp(-x))
              "softmax", #exp(x) / sum(exp(x))
              #"softplus", #log(exp(x) + 1)
              #"softsign", #x / (abs(x) + 1)
              #"tanh", #sinh(x) / cosh(x)
              #"hard_sigmoid", #0 if x < -2.5  || 1 if x > 2.5 || 0.2 * x + 0.5 if -2.5 <= x <= 2.5
              ]


i = 2

#functional
#inputs = Input(shape=(2,))
#hidden1 = Dense(units=2)(inputs) # hidden layer 1
#hidden2 = Dense(units=2,activation=activation[i])(hidden1) # hidden layer 2
#outputs = Dense(units=1,activation=outputs[i])(hidden2)
#model = Model(inputs=inputs,outputs=outputs)


#sequential
model = Sequential()
model.add(Input(shape=(2,)))
model.add(layers.Dense(units=10,activation=activation[2]))#kernel_initializer=initializers.Ones())) 
model.add(layers.Dense(units=1,activation=activation[2]))#kernel_initializer=initializers.Ones()))

model.summary()

opt = SGD(lr=0.1)
model.compile(loss='MeanSquaredError',metrics = ['accuracy'],optimizer='adam')#=opt)


ll = len(model.layers)
lw = len(model.weights)
print("numbers of layers = ", ll)
print("Number of weights after calling the model:",lw)

print()
print()

n = 6
ep = [10,20,50,100,200,300,500]
maxEpoch = ep[n]

#OR
inputs = np.asarray([[1,1], [0,0], [1,0], [0,1]])
assert not np.any(np.isnan(inputs))
expected = np.asarray([[1], [0], [1], [1]])
history = model.fit(inputs,expected,epochs=maxEpoch)

print(inputs)
print(model.predict(inputs))
test_scores = model.evaluate(inputs,expected)
print("Test loss:", test_scores)
print(history.history['accuracy'][-1])

#"Num Hidden Layers","Hidden Layer Size","ePoch","Activation Function","Accuracy"]
table.add_row([ll-1,model.layers[0].output_shape[1],maxEpoch,activation[i],test_scores[1]])
print(table)

#print("Test accuracy:", test_scores[1])



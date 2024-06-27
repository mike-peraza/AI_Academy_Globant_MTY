############################################################################################
# Example to convert Celsius (C) to Fahrenheit (F)
# F = C * 1.8 + 32

# Regular programing
# def function(C):
#     F = C * 1.8 + 32
#     return F
############################################################################################

# Step 1. Import the libraries
# pip install tensorflow
# To install a package as administrator use the --user option
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
from tensorflow.python.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# Step 2. Prepare the data.
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# Step 3. Design the model for prediction. 
# We will use keras framework to work with the Neuronal Network.
keras = tf.keras
# Setup a dense layer. Which means each node on a layer 
# will have all connections with the nodes on the next layer.
capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

# Next step, prepare the model to be compiled.
# We indicate the parameters on how the model will process the data in order to learn better.
# For the optimizer parameter, we will use the Adam alghoritm. 
# This will allow the model to adjust the weights and bais to learn and get better results.
modelo.compile(
    # Specified the learning cup
    optimizer = tf.keras.optimizers.Adam(0.1),
    # For the lost we specified the mean_squared_error function
    loss='mean_squared_error'
)

# Step 4. Training the model.
# The fit function will allow us to training the model. 
# The first two parameters are the input and output data we setup earlier.
# The third parameter is the number of loops we want the function to execute.
# The last parameter will setup the verbose flag to false to don't print all the details of the training.
print('Starting training...')
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print('Model trained')

# Step 5 Test the trained model
print('Test a prediction for 100°C...')
testCelsius = 100.0
resultado = modelo.predict(x=np.array([testCelsius]))
print('The outcome for: ' + str(testCelsius) + 'is: ' + str(resultado) + ' fahrenheit!')
# El resultado es[[211.74016]] fahrenheit!

# Step 6 (Optinal). We can print the outcome of the lost function 
# to see how bad or good the result is in each loop .
plt.xlabel('# Epoca')
plt.ylabel('Magnitud de perdida')
plt.plot(historial.history['loss'])
plt.show()

# print('Variables internas del modelo:')
# print(capa.get_weights())
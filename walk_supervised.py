import numpy as np
import pandas as pd
import time
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.callbacks import LearningRateScheduler

from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.python.client import device_lib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_layers', type=int, required=True)
parser.add_argument('--layer_size', type=int, required=True)
parser.add_argument('--activation', type=str, required=True)
parser.add_argument('--learning_rate', type=float, required=True)
parser.add_argument('--loss_function', type=str, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--drop', type=float, required=True)
args = parser.parse_args()

#Hyperparameters

start_time = time.time()

N_LAYERS = args.n_layers
LAYER_SIZE = args.layer_size
ACTIVATION = args.activation
INPUT_SIZE = 25
OUTPUT_SIZE = 22
LEARNING_RATE = args.learning_rate
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-08
LOSS_FUNCTION = args.loss_function
EPOCHS = 1000
EPOCHS_DROP = EPOCHS//10
BATCH_SIZE = args.batch_size
DROP = args.drop

#Dataset processing

print("Dataset processing...")

dataset = pd.read_csv('itandroids_walk.txt', delimiter = "::", engine = 'python')
dataset = dataset.dropna(axis=0, how='any')
train_X = dataset.values[:, :25]
train_Y = dataset.values[:, 25:]

#Neural Network Design
model =  Sequential()
model.add(Dense(LAYER_SIZE, input_dim = INPUT_SIZE, activation = ACTIVATION))
for i in range(N_LAYERS - 2):
        model.add(Dense(LAYER_SIZE, input_dim = INPUT_SIZE, activation = ACTIVATION))
model.add(Dense(OUTPUT_SIZE))


model.summary()
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=False)


#Training Procedure
def step_decay(epoch):
   initial_lrate = LEARNING_RATE
   drop = DROP
   epochs_drop = float(EPOCHS_DROP)
   lrate = initial_lrate * np.power(drop,  
           np.floor((1+epoch)/epochs_drop))
   return lrate

lrate = LearningRateScheduler(step_decay)

adam = Adam(lr=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)
model.compile (loss = LOSS_FUNCTION, optimizer = adam, metrics = ['mse', 'mae'])
history_callback = model.fit (train_X, train_Y, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = 2, callbacks=[lrate])
loss_hist_mse = np.array(history_callback.history["mean_squared_error"])
loss_hist_mae = np.array(history_callback.history["mean_absolute_error"])

filename = str(args.n_layers) + '_' + str(args.layer_size) + '_' + str(args.activation) + '_' + str(args.learning_rate) + '_' + str(args.loss_function) + '_' + str(args.batch_size) + '_' + str(args.drop)
#Saving Metrics History for plot
np.savetxt('mae_' +  filename + '.txt', loss_hist_mae, delimiter='\n')

#Save the model
print (K.get_session().graph)
model.save(filename)
elapsed_time = time.time() - start_time
with open('elapsed_time_' + filename + '.txt','w') as f: 
	f.write(str(elapsed_time))
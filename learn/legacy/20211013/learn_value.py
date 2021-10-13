import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input, concatenate, Flatten, Dropout
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, LambdaCallback
from tensorflow.keras.optimizers import Adam
#from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from random import random, randint, shuffle, sample
import subprocess
from math import exp
import datetime

hw = 8
hw2 = 64

all_data = {}

n_epochs = 20
game_num = 1000
test_ratio = 0.1
n_param = 15

kernel_size = 3
n_kernels = 50
n_residual = 1

leakyrelu_alpha = 0.01

train_param = None
train_value = None

test_raw_board = []
test_param = []
test_value = []

mean = None
std = None

my_evaluate = subprocess.Popen('./evaluation.out'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)

def digit(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n

def join_yx(y, x):
    return y * hw + x

def calc_idx(i, j, rnd):
    if rnd == 0:
        return join_yx(i, j)
    elif rnd == 1:
        return join_yx(j, hw - 1 - i)
    elif rnd == 2:
        return join_yx(hw - 1 - i, hw - 1 - j)
    else:
        return join_yx(hw - 1 - j, i)

def collect_data(num):
    global all_data
    try:
        with open('learn_data/' + digit(num, 7) + '.txt', 'r') as f:
            data = list(f.read().splitlines())
    except:
        return
    for datum in data:
        board, policy, score = datum.split()
        policy = int(policy)
        score = float(score)
        #print(board, policy, score)
        all_data.append([board, policy, score])

def reshape_data_train():
    global train_param, train_value, mean, std
    tmp_data = []
    print('calculating score & additional data')
    for itr in trange(len(all_data)):
        board, policy, score = all_data[itr]
        tmp_data.append([board, score])
    shuffle(tmp_data)
    ln = len(tmp_data)
    print('got', ln)
    print('creating train data & labels')
    train_idx = 0
    for ii in trange(ln):
        board, score = tmp_data[ii]
        my_evaluate.stdin.write(board.encode('utf-8'))
        my_evaluate.stdin.flush()
        param = [float(elem) for elem in my_evaluate.stdout.readline().rstrip().split()]
        for i in range(n_param):
            train_param[train_idx][i] = param[i]
        train_value[train_idx] = score
        train_idx += 1
    train_param = train_param[0:train_idx]
    train_value = train_value[0:train_idx]
    mean = train_param.mean(axis=0)
    std = train_param.std(axis=0)
    print('mean', mean)
    print('std', std)
    train_param = (train_param - mean) / std
    print('train', train_param.shape, train_value.shape)

def reshape_data_test():
    global test_param, test_value
    tmp_data = []
    print('calculating score & additional data')
    for itr in trange(len(all_data)):
        board, policy, score = all_data[itr]
        tmp_data.append([board, score])
    shuffle(tmp_data)
    ln = len(tmp_data)
    print('got', ln)
    print('creating train data & labels')
    for ii in trange(ln):
        board, score = tmp_data[ii]
        my_evaluate.stdin.write(board.encode('utf-8'))
        my_evaluate.stdin.flush()
        param = [float(elem) for elem in my_evaluate.stdout.readline().rstrip().split()]
        test_param.append(param)
        test_value.append(score)
    test_param = np.array(test_param)
    test_value = np.array(test_value)
    test_param = (test_param - mean) / std
    print('test', test_param.shape, test_value.shape)

def LeakyReLU(x):
    return tf.math.maximum(0.01 * x, x)


inputs = Input(shape=(n_param,))
x = Dense(16)(inputs)
x = LeakyReLU(x)
x = Dense(16)(x)
x = LeakyReLU(x)
x = Dense(16)(x)
x = LeakyReLU(x)
y = Dense(1)(x)
y = Activation('tanh')(y)

model = Model(inputs=inputs, outputs=y)

model.summary()

print('collecting data')

n_train_data = int(game_num * (1.0 - test_ratio))
n_test_data = int(game_num * test_ratio)

idxes = list(range(game_num + 100))
shuffle(idxes)


all_data = []
for i in trange(n_train_data):
    collect_data(idxes[i])
train_param = np.zeros((len(all_data), n_param))
train_value = np.zeros(len(all_data))
reshape_data_train()

all_data = []
for i in trange(n_train_data, game_num):
    collect_data(idxes[i])
test_raw_board = []
test_param = []
test_value = []
reshape_data_test()

model.compile(loss='mse', optimizer='adam')

print(model.evaluate(train_param, train_value))
early_stop = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(train_param, train_value, epochs=n_epochs, validation_data=(test_param, test_value), callbacks=[early_stop])

with open('param/param_value.txt', 'w') as f:
    i = 0
    while True:
        try:
            #print(i, model.layers[i])
            dammy = model.layers[i]
            j = 0
            while True:
                try:
                    print(model.layers[i].weights[j].shape)
                    if len(model.layers[i].weights[j].shape) == 4:
                        for ll in range(model.layers[i].weights[j].shape[3]):
                            for kk in range(model.layers[i].weights[j].shape[2]):
                                for jj in range(model.layers[i].weights[j].shape[1]):
                                    for ii in range(model.layers[i].weights[j].shape[0]):
                                        f.write('{:.14f}'.format(model.layers[i].weights[j].numpy()[ii][jj][kk][ll]) + '\n')
                    elif len(model.layers[i].weights[j].shape) == 2:
                        for ii in range(model.layers[i].weights[j].shape[0]):
                            for jj in range(model.layers[i].weights[j].shape[1]):
                                f.write('{:.14f}'.format(model.layers[i].weights[j].numpy()[ii][jj]) + '\n')
                    elif len(model.layers[i].weights[j].shape) == 1:
                        for ii in range(model.layers[i].weights[j].shape[0]):
                            f.write('{:.14f}'.format(model.layers[i].weights[j].numpy()[ii]) + '\n')
                    j += 1
                except:
                    break
            i += 1
        except:
            break
now = datetime.datetime.today()
print(str(now.year) + digit(now.month, 2) + digit(now.day, 2) + '_' + digit(now.hour, 2) + digit(now.minute, 2))
#model.save('param/additional_learn_model/' + str(now.year) + digit(now.month, 2) + digit(now.day, 2) + '_' + digit(now.hour, 2) + digit(now.minute, 2) + '.h5')
model.save('param/model_value.h5')

for key in ['loss', 'val_loss']:
    plt.plot(history.history[key], label=key)
plt.xlabel('epoch')
plt.ylabel('value loss')
plt.legend(loc='best')
plt.savefig('graph/value_net_loss.png')
plt.clf()

all_data = []
for i in trange(game_num, game_num + 100):
    collect_data(idxes[i])
test_raw_board = []
test_param = []
test_value = []
reshape_data_test()

print(model.evaluate(test_param, test_value))

prediction = model.predict(test_param[:10])
for i in range(10):
    print(test_raw_board[i])
    print(prediction[i])

my_evaluate.kill()
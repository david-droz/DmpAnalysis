'''

Plot the training history of a Keras run.

'''

from __future__ import print_function, division, absolute_import

import numpy as np
import pickle
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_hdf('./results/'+sys.argv[1]+'/history.hdf')

fig1 = plt.figure()
plt.plot(df.index.tolist(),np.asarray(df["acc"]),'-',label='acc')
plt.plot(df.index.tolist(),np.asarray(df["val_acc"]),'-',label='val_acc')
plt.plot(df.index.tolist(),np.asarray(df["loss"]),'-',label='loss')
plt.plot(df.index.tolist(),np.asarray(df["val_loss"]),'-',label='val_loss')
plt.legend(loc='best')
plt.title("Training history")
plt.ylabel('classification accuracy')
plt.xlabel('epoch')

plt.savefig('history')

import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten, Dropout, MaxPooling2D
from keras.optimizers import SGD
import tensorflow as tf

import pandas_ml as pdml
import imblearn

df = pd.read_csv('creditcard.csv', low_memory=False)

# print(df.head())
#data bersifat anonim karena mengandung hal sensitif
#data dari kaggle berdasar pada transaksi kartu kredit selama 2 hari di Eropa pada bulan September 2013
#keterangan dari kaggle class memiliki 2 buah nilai, 1 untuk fraud (penipuan) dan 0 untuk non fraud (non-penipuan)
#ammount berisikan nilai transaksi, digunakan sebagai nilai x karena kita mengetahui secara jelas data apa yang ada dalam kolom tersebut(tidak anonim)

X = df.iloc[:,:-1]
y = df['Class']

frauds = df.loc[df['Class'] == 1]
non_frauds = df.loc[df['Class'] == 0]
print("Ada ", len(frauds), " data penipuan dan ", len(non_frauds), " data non-penipuan.")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

print("Jumlah data yang di training: ", X_train.shape)

model = Sequential()
model.add(Dense(30, input_dim=30, activation='relu'))     # kernel_initializer='normal'
model.add(Dense(1, activation='sigmoid'))                 # kernel_initializer='normal'
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train.as_matrix(), y_train, epochs=1)

print("Loss: ", model.evaluate(X_test.as_matrix(), y_test, verbose=0))

y_predicted = model.predict(X_test.as_matrix()).T[0].astype(int)

from pandas_ml import ConfusionMatrix
y_right = np.array(y_test)
confusion_matrix = ConfusionMatrix(y_right, y_predicted)
print("Confusion matrix:\n%s" % confusion_matrix)
confusion_matrix.plot(normalized=True)
plt.show()
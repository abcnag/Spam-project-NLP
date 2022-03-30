import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM , Activation , Dropout , Dense , Embedding , Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
#####
data = pd.read_csv('C:/Users/M/Desktop/Programming/spam.csv')
data.Category.replace({'ham' : 0 , 'spam' : 1},inplace=True)
#####
x_train , x_test , y_train , y_test = train_test_split(data.Message,data.Category,test_size=0.2)
y_train = np.asarray([i for i in y_train])
y_test = np.asarray([i for i in y_test])
#####
# Convert messages to their words indeces
tkn = Tokenizer(10000)
tkn.fit_on_texts(x_train)
seq_test = tkn.texts_to_sequences(x_test)
seq_train = tkn.texts_to_sequences(x_train)

# Padding
max_len = 100
seq_test = pad_sequences(seq_test , padding="pre" , maxlen=max_len)
seq_train = pad_sequences(seq_train , padding="pre" , maxlen=max_len)
###
# Creat model 
Model = Sequential()
Model.add(Embedding(input_dim=10000 , output_dim=30 , input_length=max_len))
Model.add(Dropout(0.01))
Model.add(LSTM(256))
Model.add(Dropout(0.01))
Model.add(Dense(1,activation='sigmoid'))

# Compile model
Model.compile(loss='binary_crossentropy' , optimizer='adam' , metrics=['accuracy'])

# Train model
Model.fit(seq_train , y_train , validation_data=(seq_test , y_test) , epochs=3
)

# Accuracy
accuracy = 0
pre = Model.predict(seq_test)
for i in range(len(pre)):
    if (pre[i] >= 0.5 and y_test[i] == 1) or (pre[i] < 0.5 and y_test[i] == 0):
        accuracy += 1
print('Model accuracy :' , accuracy/len(pre))

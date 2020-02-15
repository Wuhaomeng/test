from keras.layers import Dense, Flatten, Input
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
import numpy as np

# define documents
docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!',
        'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']

# define class labels
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
# integer encode the documents
vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]   #one_hot编码到[1,n],不包括0
print(encoded_docs)

# pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)

# define the model
input = Input(shape=(4, ))     #(4,)对应(*,4)的情况
x = Embedding(vocab_size, 8, input_length=max_length)(input)    #这一步对应的参数量为50*8
x = Flatten()(x)    #按正常顺序降维
x = Dense(1, activation='sigmoid')(x)   #全连接神经网络层
model = Model(inputs=input, outputs=x)
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())

# fit the model
model.fit(padded_docs, labels, epochs=100, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)#评估模型，不输出测试结果
loss_test, accuracy_test = model.evaluate(padded_docs, labels, verbose=0) #verbose为日志显示，verbose=0为不在标准输出流输出日志信息，verbose=1为输出进度条记录，verbose=2为每个epoch输出一条记录
print('Accuracy: %f' % (accuracy * 100))
# test the model
test = one_hot('good',50)
padded_test = pad_sequences([test], maxlen=max_length, padding='post')
print(model.predict(padded_test))#输出测试结果

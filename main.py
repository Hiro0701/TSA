import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding

df = pd.read_csv('/Users/chrr/PycharmProjects/TSA/Tweets.csv')

print(df.head(10))
print(df.columns)

review_df = df[['text', 'airline_sentiment']]
review_df = review_df[review_df['airline_sentiment'] != 'neutral']

print(review_df['airline_sentiment'].value_counts())

sentiment_label = review_df.airline_sentiment.factorize()
print(sentiment_label)

tweet = review_df.text.values
print(tweet)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(tweet)
encoded_docs = tokenizer.texts_to_sequences(tweet)

print(encoded_docs)
print(tokenizer.word_index)

padded_sequence = pad_sequences(encoded_docs, maxlen=200)

print(padded_sequence)

X_train, X_test, y_train, y_test = train_test_split(padded_sequence, sentiment_label[0], test_size=0.2, random_state=42)

vocab_size = len(tokenizer.word_index) + 1
embedding_vector_length = 32

model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length, input_length=200))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

history = model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=32)

test_loss, test_accuracy = model.evaluate(X_test, y_test)

plt.plot(history.history['accuracy'], label='acc', color='r')
plt.plot(history.history['val_accuracy'], label='val_acc', color='b')
plt.axhline(y=test_accuracy, label='test_acc', color='g')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='loss', color='r')
plt.plot(history.history['val_loss'], label='val_loss', color='b')
plt.axhline(y=test_loss, label='test_loss', color='g')
plt.legend()
plt.show()

my_text = "Worst flight ever."
my_token = tokenizer.texts_to_sequences([my_text])
my_seq = pad_sequences(my_token, maxlen=200)
prediction = model.predict(my_seq)

model.save('TSA.h5')

with open('token.json', 'w') as f:
    json.dump(Tokenizer.to_json(tokenizer), f)

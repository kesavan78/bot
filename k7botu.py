import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load your dataset from dialgoue.txt or any other source
with open("dialogs.txt", "r", encoding="utf-8") as file:
    data = file.read()

# Tokenize the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data.split("\n"))
total_words = len(tokenizer.word_index) + 1

# Create input sequences and target sequences
input_sequences = []
for line in data.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_length = max([len(seq) for seq in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre'))

# Create predictors and labels
predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
label = tf.keras.utils.to_categorical(label, num_classes=total_words)

# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(total_words, 100, input_length=max_sequence_length - 1))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)))
model.add(tf.keras.layers.Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(predictors, label, epochs=100, verbose=1)

# Chat with the bot
def chat_with_bot(text, model, tokenizer, max_sequence_length):
    for _ in range(3):  # Number of responses to generate
        input_text = text
        for _ in range(10):  # Maximum number of tokens in the response
            token_list = tokenizer.texts_to_sequences([input_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
            predicted = model.predict_classes(token_list, verbose=0)
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            input_text += " " + output_word
            print("Chatbot:", input_text)

# Example conversation
chat_with_bot("Hello", model, tokenizer, max_sequence_length)

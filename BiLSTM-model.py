import csv
from collections import defaultdict
import numpy as np
import json
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, TimeDistributed
from keras.optimizers import SGD
from keras.utils import pad_sequences, to_categorical
from keras import backend as K
from sklearn.metrics import f1_score

DESIRED_LENGTH = 200
START_TAG = "<START>"
STOP_TAG = "<STOP>"

def load_from_csv(file_path):
    data = defaultdict(list)
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data[int(row['sent_id'])].append((row['word'], row['tag']))
    return list(data.values())

def create_mappings(sentences):
    word_to_id = {"<PAD>": 0, "<UNK>": 1}
    tag_to_id = {"<PAD>": 0, START_TAG: 1, STOP_TAG: 2}
    for sentence in sentences:
        for word, tag in sentence:
            if word not in word_to_id:
                word_to_id[word] = len(word_to_id)
            if tag not in tag_to_id:
                tag_to_id[tag] = len(tag_to_id)
    return word_to_id, tag_to_id

def convert_to_ids(sentences, word_to_id, tag_to_id):
    sentences_ids = []
    tags_ids = []
    for sentence in sentences:
        sentence_ids = [word_to_id.get(word, word_to_id["<UNK>"]) for word, _ in sentence]
        tag_ids = [tag_to_id[tag] for _, tag in sentence]
        sentences_ids.append(sentence_ids)
        tags_ids.append(tag_ids)
    return sentences_ids, tags_ids

def pad_sequences_with_tags(sequences, tags, max_length, tag_to_id):
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    padded_tags = pad_sequences(tags, maxlen=max_length, padding='post', truncating='post', value=tag_to_id[STOP_TAG])
    return padded_sequences, padded_tags

def create_weighted_loss(class_weights):
    def weighted_loss(y_true, y_pred):
        loss = K.categorical_crossentropy(y_true, y_pred)
        weights = K.sum(class_weights * y_true, axis=-1)
        loss = loss * weights
        return loss
    return weighted_loss

def build_model(vocab_size, tag_to_id, embedding_dim, hidden_dim, dropout_rate, learning_rate):
    input_layer = Input(shape=(DESIRED_LENGTH,))
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_layer)
    bilstm_layer1 = Bidirectional(LSTM(hidden_dim // 2, return_sequences=True))(embedding_layer)
    dropout_layer1 = Dropout(dropout_rate)(bilstm_layer1)
    bilstm_layer2 = Bidirectional(LSTM(hidden_dim // 2, return_sequences=True))(dropout_layer1)
    dropout_layer2 = Dropout(dropout_rate)(bilstm_layer2)
    output_layer = TimeDistributed(Dense(len(tag_to_id), activation='softmax'))(dropout_layer2)
    optimizer = SGD(learning_rate=learning_rate)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate, class_weights):
    weighted_loss = create_weighted_loss(class_weights)
    model.compile(loss= weighted_loss, optimizer=SGD(learning_rate=learning_rate), 
                  metrics=['accuracy'])
    for epoch in range(epochs):
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=batch_size, verbose=1)
    return model

def calc_f1(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    return f1_score(y_true_flat, y_pred_flat, average='macro')

def save_model(model, path):
    model.save(path)

def save_results(results, path):
    with open(path, 'w') as file:
        json.dump(results, file)

def main():
    train_file = '/content/drive/MyDrive/data/train1.csv'
    val_file = '/content/drive/MyDrive/data/val1.csv'
    test_file = '/content/drive/MyDrive/data/test1.csv'

    train_sentences = load_from_csv(train_file)
    val_sentences = load_from_csv(val_file)
    test_sentences = load_from_csv(test_file)
    word_to_id, tag_to_id = create_mappings(train_sentences)
    X_train, y_train = convert_to_ids(train_sentences, word_to_id, tag_to_id)
    X_val, y_val = convert_to_ids(val_sentences, word_to_id, tag_to_id)
    X_test, y_test = convert_to_ids(test_sentences, word_to_id, tag_to_id)
    padded_X_train, padded_y_train = pad_sequences_with_tags(X_train, y_train, DESIRED_LENGTH, tag_to_id)
    padded_X_val, padded_y_val = pad_sequences_with_tags(X_val, y_val, DESIRED_LENGTH, tag_to_id)
    padded_X_test, padded_y_test = pad_sequences_with_tags(X_test, y_test, DESIRED_LENGTH, tag_to_id)
    vocab_size = len(word_to_id)

    hidden_dim_values = [20, 50, 100, 200]
    embedding_dim_values = [100, 200, 2000, 4000]
    dropout_rate_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    learning_rate_values = [0.0001, 0.01, 0.1, 0.2, 0.5]
    batch_size_values = [8, 32, 128]
    EPOCHS = 1

    best_f1 = 0.0
    best_model = None

    for hidden_dim in hidden_dim_values:
        for embedding_dim in embedding_dim_values:
            for dropout_rate in dropout_rate_values:
                for learning_rate in learning_rate_values:
                    for batch_size in batch_size_values:
                        model = build_model(vocab_size, tag_to_id, embedding_dim, hidden_dim, dropout_rate, learning_rate)
                        class_weights = [0.0001, 0.0001, 0.0001, 1.31, 38.26, 46.43]
                        model = train_model(model, padded_X_train, to_categorical(padded_y_train),
                                                     padded_X_val, to_categorical(padded_y_val),
                                                     EPOCHS, batch_size, learning_rate, class_weights)
                        val_preds = model.predict(padded_X_val)
                        val_preds = np.argmax(val_preds, axis=-1)
                        val_f1 = calc_f1(padded_y_val, val_preds)
                        if val_f1 > best_f1:
                            best_f1 = val_f1
                            best_model = model
                            best_hyperparameters = {'hidden_dim': hidden_dim,
                                                    'embedding_dim': embedding_dim,
                                                    'dropout_rate': dropout_rate,
                                                    'learning_rate': learning_rate,
                                                    'batch_size': batch_size}

    if best_model is not None:
        save_model(best_model, 'best_model.h5')
        save_results({'word_to_id': word_to_id, 'tag_to_id': tag_to_id}, 'results.json')
        test_preds = best_model.predict(padded_X_test)
        test_preds = np.argmax(test_preds, axis=-1)
        test_f1 = calc_f1(padded_y_test, test_preds)
        save_results({'test_f1': test_f1, 'hyperparameters': best_hyperparameters}, 'test_results.json')

if __name__ == '__main__':
    main()

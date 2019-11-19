import re
from pickle import load, dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding
from numpy import array
from random import randint
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


def load_text(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


def clean_text(text):
    text = text.replace('?', ' ? ').replace('!', ' ! ').replace('৷', ' ৷ ').replace('।', ' । ')
    text = re.sub(u'[০-৯]+', '', text)

    text = re.sub('[ \t°`\'\"‘’“”()♦►\-–,:\.;—\n/]+', ' ', text)
    # text = re.sub('।৷', ' ', text)
    # punc = str.maketrans('', '', string.punctuation)
    tokens = [w for w in text.split() if len(w) > 0]
    # tokens = [w.translate(punc) for w in tokens]
    # tokens = [w for w in tokens if w.isalpha()]
    return tokens


def save_text(lines, filename):
    text = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(text)
    file.close()


def generate_sequences(input_file, seq_length):
    text = load_text(input_file)
    tokens = clean_text(text)
    print('Total Tokens: %d' % len(tokens))
    print('Unique Tokens: %d' % len(set(tokens)))

    length = seq_length + 1  # extra one for target
    sequences = []
    for i in range(len(tokens) - length + 1):
        seq = tokens[i:i + length]
        line = ' '.join(seq)
        sequences.append(line)
    print('Total sequences {0}'.format(len(sequences)))

    out_filename = 'sequences.txt'
    save_text(sequences, out_filename)
    return out_filename


def create_model(vocab_size, input_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 20, input_length=input_length))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_model(lines, seq_length, batch_size=128, epochs=100):
    # integer encode sequences of words
    # default filters '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n'
    # but i want to keep line ender punctuations
    tokenizer = Tokenizer(filters='"#$%&()*+,-./:;<=>@[\]^_`{}~\t\n')
    tokenizer.fit_on_texts(lines)
    sequences = tokenizer.texts_to_sequences(lines)

    # vocabulary size
    vocab_size = len(tokenizer.word_index) + 1

    # separate into input and output

    sequences = array(sequences)
    length = seq_length + 1
    for index, seq in enumerate(sequences):
        if len(seq) != length:
            print('Index ', index, ' Length ', len(seq))
            return None, None

    X, y = sequences[:, :-1], sequences[:, -1]
    y = to_categorical(y, num_classes=vocab_size)
    input_length = X.shape[1]

    model = create_model(vocab_size, input_length)

    file_path = 'best_model.h5'
    checkpoint = ModelCheckpoint(file_path, monitor='acc', verbose=1, save_best_only=True, mode='max')
    # fit model
    model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint])

    return model, tokenizer


def save_model(model, tokenizer, model_name='model.h5', tokenizer_name='tokenizer.pkl'):
    model.save(model_name)
    dump(tokenizer, open(tokenizer_name, 'wb'))


def load_model_tokenizer(model_name, tokenizer_name):
    model = load_model(model_name)
    tokenizer = load(open(tokenizer_name, 'rb'))
    return model, tokenizer


def test_model(in_filename, model, tokenizer, n_words=10, seed_text=''):
    # load cleaned text sequences
    text = load_text(in_filename)
    lines = text.split('\n')
    seq_length = len(lines[0].split()) - 1

    # select a seed text if not set
    if len(seed_text) == 0:
        seed_text = lines[randint(0, len(lines))]
    # print('Input: ', seed_text +'\n')
    seed_texts = seed_text.split()
    print('Input :' + ' '.join(seed_texts[:5]) + '\n' + ' '.join(seed_texts[5:]) + '\n')

    # generate new text
    generated = generate_output(model, tokenizer, seq_length, seed_text, n_words)

    words = generated.split()
    step = 7
    print('AI output :')
    for i in range(0, n_words, step):
        output = words[i: min(n_words, i + step)]
        print(' '.join(output))
    # print('Generated ', generated)


def generate_output(model, tokenizer, seq_length, seed_text, n_words):
    in_text = seed_text
    # word_to_index = {index:word for word,index in tokenizer.word_index.items()}
    result = ''
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # pre-pad sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        y_pred = model.predict_classes(encoded, verbose=0)

        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == y_pred:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
        result += ' ' + out_word
    return result
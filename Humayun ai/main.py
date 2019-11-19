from Util import *

input_file = 'Dorjar Opashe - Himu.txt'
seq_length = 10
seq_file = generate_sequences(input_file, seq_length)

lines = load_text(seq_file).split('\n')
model, tokenizer = train_model(lines, seq_length, batch_size=128, epochs=2)
dump(tokenizer, open('tekenizer.pkl', 'wb'))
best_model, tokenizer = load_model_tokenizer('best_model.h5', 'tokenizer.pkl')

seq_file = 'sequences.txt'
test_model(seq_file, best_model, tokenizer, n_words=70, seed_text=lines[0])
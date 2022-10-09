# # ## 8.1 Introduction to Deep Learning for NLP
# # ## Nothing to implement :)
# #
# #
# # ## 8.2 The Basic Perceptron Model
# # ## Nothing to implement :)
# #
# #
# # ## 8.3 Introduction to Neural Networks
# # ## Nothing to implement :)
# #
# #
# # ## 8.4 Keras Basics - Part One + 8.5 Keras Basics - Part Two
# # import numpy as np
# # from sklearn.datasets import load_iris
# # iris = load_iris()
# # print(type(iris))
# # print(iris.DESCR)
# #
# # X = iris.data
# # print(X)
# #
# # y = iris.target
# # print(y)
# #
# # # WE HAVE TO PRODUCE ONE_HOTS
# # from keras.utils import to_categorical
# # y = to_categorical(y)
# # print(y.shape)
# # print(y)
# #
# # from sklearn.model_selection import train_test_split
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# #
# # from sklearn.preprocessing import MinMaxScaler
# # scaler_obj = MinMaxScaler()
# #
# # scaler_obj.fit(X_train)
# # scaled_X_train = scaler_obj.transform(X_train)
# #
# # scaler_obj.fit(X_test)
# # scaled_X_test = scaler_obj.transform(X_test)
# #
# # from keras.models import Sequential
# # from keras.layers import Dense
# # model = Sequential()
# # model.add(Dense(units=8, input_dim=4, activation='relu'))
# # model.add(Dense(units=8, input_dim=8, activation='relu'))
# # model.add(Dense(units=3, activation='softmax')) # an output should looks like [0.1, 0.2, 0.7]
# # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# # print(model.summary())
# #
# # model.fit(scaled_X_train, y_train, epochs=150, verbose=2)
# #
# # predictions_onehot_style = model.predict(scaled_X_test)
# # predictions = model.predict_classes(scaled_X_test)
# #
# # from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# # print(confusion_matrix(y_test.argmax(axis=1), predictions))
# # print(classification_report(y_test.argmax(axis=1), predictions))
# # print(accuracy_score(y_test.argmax(axis=1), predictions))
# #
# # model.save('./trained_models/my_first_model.h5')
# # from keras.models import load_model
# # new_model = load_model('./trained_models/my_first_model.h5')
# # print(new_model.predict_classes(scaled_X_test))
# #
# #
# # ## 8.6 Recurrent Neural Network Overview
# # ## Nothing to implement :)
# #
# #
# # ## 8.7 LSTMs, GRU, and Text Generation
# # ## Nothing to implement :)
# #
# #
# # ## 8.8 Text Generation with LSTMs with Keras and Python - Part One
# # def read_file(file_path):
# #     with open(file_path) as f:
# #         str_text = f.read()
# #     return str_text
# #
# # import spacy
# # nlp = spacy.load('en_core_web_lg', disable=['parser', 'tagger', 'ner'])
# # nlp.max_lenght = 1198623
# #
# # def separate_punc(doc_text):
# #     return [token.text.lower() for token in nlp(doc_text) if token.text not in \
# #             '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n ']
# #
# # d = read_file('./files_for_practice/moby_dick_four_chapters.txt')
# # tokens = separate_punc(d)
# # print(tokens)
# # print(len(tokens))
# #
# # ## WE ARE GOING TO GIVE 25 WORDS --> OUT NETWORK PREDICT 26th WORD
# # train_len = 25 + 1
# # text_sequences = []
# # for i in range(train_len, len(tokens)):
# #     seq = tokens[i-train_len:i]
# #     text_sequences.append(seq)
# # print(type(text_sequences))
# # print(text_sequences[0])
# # print(len(text_sequences[1]))
# #
# # from keras.preprocessing.text import Tokenizer
# # tokenizer = Tokenizer()
# # tokenizer.fit_on_texts(text_sequences)
# # sequences = tokenizer.texts_to_sequences(text_sequences)
# #
# # print(sequences[0])
# # print(len(sequences[1]))
# # print(tokenizer.index_word)
# #
# # for i in sequences[0]:
# #     print(f"{i} : {tokenizer.index_word[i]}")
# #
# # print(tokenizer.word_counts)
# # vocabulary_size = len(tokenizer.word_counts)
# # print(vocabulary_size)
# #
# # import numpy as np
# # sequences = np.array(sequences)
# # print(sequences)
# #
# #
# # ## 8.9 Text Generation with LSTMs with Keras and Python - Part Two
# # def read_file(file_path):
# #     with open(file_path) as f:
# #         str_text = f.read()
# #     return str_text
# #
# # import spacy
# # nlp = spacy.load('en_core_web_lg', disable=['parser', 'tagger', 'ner'])
# # nlp.max_lenght = 1198623
# #
# # def separate_punc(doc_text):
# #     return [token.text.lower() for token in nlp(doc_text) if token.text not in \
# #             '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n ']
# #
# # d = read_file('./files_for_practice/moby_dick_four_chapters.txt')
# # tokens = separate_punc(d)
# #
# # # WE ARE GOING TO GIVE 25 WORDS --> OUT NETWORK PREDICT 26th WORD
# # train_len = 25 + 1
# # text_sequences = []
# # for i in range(train_len, len(tokens)):
# #     seq = tokens[i-train_len:i]
# #     text_sequences.append(seq)
# #
# # from keras.preprocessing.text import Tokenizer
# # tokenizer = Tokenizer()
# # tokenizer.fit_on_texts(text_sequences)
# # sequences = tokenizer.texts_to_sequences(text_sequences)
# # vocabulary_size = len(tokenizer.word_counts)
# # print(vocabulary_size)
# #
# # import numpy as np
# # sequences = np.array(sequences)
# #
# # from keras.utils import to_categorical
# # X = sequences[:, :-1]
# # y = sequences[:, -1]
# # y = to_categorical(y, num_classes=vocabulary_size+1)
# # seq_lenght = X.shape[1]
# #
# # print(X.shape)
# # print(y.shape)
# #
# # from keras.models import Sequential
# # from keras.layers import Dense, LSTM, Embedding
# #
# # def create_model(vocab_size, seq_len):
# #     model = Sequential()
# #     model.add(Embedding(input_dim=vocab_size, output_dim=seq_len, input_length=seq_len))
# #     model.add(LSTM(seq_len*6, return_sequences=True))
# #     model.add(LSTM(seq_len*6))
# #     model.add(Dense(seq_len*6, activation='relu'))
# #
# #     model.add(Dense(vocab_size, activation='softmax'))
# #
# #     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# #
# #     print(model.summary())
# #     return model
# #
# # model = create_model(vocabulary_size+1, seq_lenght)
# #
# # from pickle import dump, load
# # model.fit(X,y, batch_size=128, epochs=2, verbose=1)
# #
# # model.save('./trained_models/my_mobydick_model.h5')
# # dump(tokenizer, open('./trained_models/my_simpletokenizer', 'wb'))
# #
# #
# # ## 8.10 Text Generation with LSTMS with Keras - Part Three
# # import numpy as np
# # from pickle import dump, load
# # from keras.models import load_model
# # import spacy
# #
# # nlp = spacy.load('en_core_web_lg', disable=['parser', 'tagger', 'ner'])
# # nlp.max_lenght = 1198623
# # model = load_model('./trained_models/my_mobydick_model.h5')
# # tokenizer = load(open('./trained_models/my_simpletokenizer', mode='rb'))
# #
# # def read_file(file_path):
# #     with open(file_path) as f:
# #         str_text = f.read()
# #     return str_text
# #
# # def separate_punc(doc_text):
# #     return [token.text.lower() for token in nlp(doc_text) if token.text not in \
# #             '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n ']
# #
# # d = read_file('./files_for_practice/moby_dick_four_chapters.txt')
# # tokens = separate_punc(d)
# #
# # train_len = 25 + 1
# # text_sequences = []
# # for i in range(train_len, len(tokens)):
# #     seq = tokens[i-train_len:i]
# #     text_sequences.append(seq)
# #
# # sequences = tokenizer.texts_to_sequences(text_sequences)
# # sequences = np.array(sequences)
# # vocabulary_size = len(tokenizer.word_counts)
# #
# # seq_lenght = sequences.shape[1] - 1
# #
# # from keras.preprocessing.sequence import pad_sequences
# # def generate_text(model, tokenizer, seq_len, seed_text, num_gen_words):
# #     output_text = []
# #     input_text = seed_text
# #     for i in range(num_gen_words):
# #         encoded_text = tokenizer.texts_to_sequences([input_text])[0]
# #         pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
# #         pred_word_ind = model.predict_classes(pad_encoded, verbose=0)[0]
# #         pred_word = tokenizer.index_word[pred_word_ind]
# #         input_text = ' ' + pred_word
# #         output_text.append(pred_word)
# #     return ' '.join(output_text)
# #
# # import random
# # random.seed(101)
# # random_pick = random.randint(0, len(text_sequences))
# # random_seq_text = text_sequences[random_pick][:-1]
# # seed_text = ' '.join(random_seq_text)
# #
# # print(generate_text(model, tokenizer, seq_lenght, seed_text, num_gen_words=25))
# #
# # # # ALSO CAN USE OTHER PRETRAINED MODEL AND TOKENIZER THAT ARE MORE POWERFUL FOR THIS STAGE
# # # model = load_model('./trained_models/epochBIG.h5')
# # # tokenizer = load(open('./trained_models/epochBIG', mode='rb'))
# # # print(generate_text(model, tokenizer, seq_lenght, seed_text, num_gen_words=25))
# #
# #
# # ## 8.11 Chat Bots Overview
# # ## Nothing to implement :)
# #
# #
# # ## 8.12 Creating Chat Bots with Python - Part One + 8.13 Creating Chat Bots with Python - Part Two
# # ## + 8.14 Creating Chat Bots with Python - Part Three + 8.15 Creating Chat Bots with Python - Part Four
# import pickle
# import numpy as np
#
# with open('./files_for_practice/train_qa.txt', mode='rb') as f:
#     train_data = pickle.load(f)
#
# with open('./files_for_practice/test_qa.txt', mode='rb') as f:
#     test_data = pickle.load(f)
#
# all_data = test_data + train_data
# vocab = set()
# for story, question, answer in all_data:
#     vocab = vocab.union(set(story))
#     vocab = vocab.union(set(question))
# vocab.add('no')
# vocab.add('yes')
# vocab_len = len(vocab) + 1
#
# # FIND THE LONGEST STORY TO DESIGN THE NETWORK
# all_stories_len = [len(data[0]) for data in all_data]
# max_story_len = max(all_stories_len)
# all_questions_len = [len(data[1]) for data in all_data]
# max_question_len = max(all_questions_len)
#
# from keras.preprocessing.sequence import pad_sequences
# from keras.preprocessing.text import Tokenizer
# tokenizer = Tokenizer(filters=[])
# tokenizer.fit_on_texts(vocab)
#
#
# def vectorize_stories(data, word_index=tokenizer.word_index, max_story_len=max_story_len, max_question_len=max_question_len):
#     # STORIES = X
#     X = []
#     # QUESTIONS = Xq
#     Xq = []
#     # Y CORRECT ANSWER (yes/no)
#     Y = []
#     for story, question, answer in data:
#         x = [word_index[word.lower()] for word in story]
#         xq = [word_index[word.lower()] for word in question]
#         y = np.zeros(len(word_index) + 1)
#         y[word_index[answer]] = 1
#
#         X.append(x)
#         Xq.append(xq)
#         Y.append(y)
#
#     return (pad_sequences(X, maxlen=max_story_len),
#             pad_sequences(Xq, maxlen=max_question_len),
#             np.array(Y))
#
#
# inputs_train, queries_train, answers_train = vectorize_stories(train_data)
# inputs_test, queries_test, answers_test = vectorize_stories(test_data)
#
# from keras.layers.embeddings import Embedding
# from keras.models import Model, Sequential, load_model
# from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate, LSTM
#
# # PLACEHOLDER shape=(max_story_len, batch_size)
# input_sequence = Input((max_story_len, ))
# question = Input((max_question_len, ))
#
# vocab_size = len(vocab) + 1
#
# # INPUT ENCODER M --> the output would in the form of : (samples, story_maxlen, embedding_dim)
# input_encoder_m = Sequential()
# input_encoder_m.add(Embedding(input_dim=vocab_size, output_dim=64))
# input_encoder_m.add(Dropout(0.3))
#
# # INPUT ENCODER c --> the output would in the form of : (samples, story_maxlen, max_question_len)
# input_encoder_c = Sequential()
# input_encoder_c.add(Embedding(input_dim=vocab_size, output_dim=max_question_len))
# input_encoder_c.add(Dropout(0.3))
#
# # QUESTION ENCODER --> the output would in the form of : (samples, query_maxlen, embedding_dim)
# question_encoder = Sequential()
# question_encoder.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=max_question_len))
# question_encoder.add(Dropout(0.3))
#
# input_encoded_m = input_encoder_m(input_sequence)
# input_encoded_c = input_encoder_c(input_sequence)
# question_encoded = question_encoder(question)
#
# match = dot([input_encoded_m, question_encoded], axes=(2, 2))
# match = Activation('softmax')(match)
#
# response = add([match, input_encoded_c])
# response = Permute((2, 1))(response)
# answer = concatenate([response, question_encoded])
#
# answer = LSTM(32)(answer)
# answer = Dropout(0.5)(answer)
# answer = Dense(vocab_size)(answer)  # THE OUTPUT WOULD BE IN SHAPE (samples, vocab_size)
#
# answer = Activation('softmax')(answer)
# model = Model([input_sequence, question], answer)
#
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# print(model.summary())
#
# history = model.fit([inputs_train, queries_train], answers_train, batch_size=32, epochs=100,
#                     validation_data=([inputs_test, queries_test], answers_test))
#
# import matplotlib.pyplot as plt
# print(history.history.keys())
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'eval'], loc='upper left')
# plt.show()
#
# model.save('./trained_models/my_chatbot_model.h5')
# pickle.dump(tokenizer, open('./trained_models/my_chatbottokenizer', 'wb'))
#
# pred_result = model.predict(([inputs_test, queries_test]))
#
# # NOW TRY TO LOOK AT THE QUESTION AND ANSWER FOR ONE OF THE TEST DATA (THE FIRST ONE)
# print(' '.join(test_data[0][0]))
# print(' '.join(test_data[0][1]))
# print(test_data[0][2])
#
# val_max = np.argmax(pred_result[0])
# print(pred_result[0][val_max])
# for key, val in tokenizer.word_index.items():
#     if val == val_max:
#         print(key)
#
# # NOW TRY TO CREATE A NEW SET OF STORY, QUESTION AND ANSWER RESTRICTED TO THE VOCABULARY ON WHICH THE MODEL GOT TRAINED
# my_story = 'John left the kitchen . Sandra dropped the football in the garden .'
# my_question = 'Is the football in the garden ?'
# my_data = [(my_story.split(), my_question.split(), 'yes')]
#
# my_story, my_ques, my_ans = vectorize_stories(my_data)
#
# my_pred_result = model.predict(([my_story, my_ques]))
# my_val_max = np.argmax(my_pred_result[0])
# print(my_pred_result[0][my_val_max])
# for key, val in tokenizer.word_index.items():
#     if val == my_val_max:
#         print(key)

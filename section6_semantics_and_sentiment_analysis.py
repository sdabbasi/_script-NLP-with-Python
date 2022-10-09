# ## 6.1 Introduction to Semantics and Sentiment Analysis
# ## Nothing to implement :)
#
#
# ## 6.2 Overview of Semantics and Word Vectors
# ## Nothing to implement :)
#
#
# ## 6.3 Semantics and Word Vectors with Spacy
# import spacy
# nlp = spacy.load('en_core_web_lg')
#
# print(nlp(u'lion').vector)
# print(nlp(u'lion').vector.shape)
#
# print(nlp(u'The quick brown fox jumped').vector)
# print(nlp(u'The quick brown fox jumped').vector.shape)
#
# tokens = nlp(u'lion cat pet')
# for token1 in tokens:
#     for token2 in tokens:
#         print(token1.text, token2.text, token1.similarity(token2))
#
# tokens = nlp(u'like love hate')
# for token1 in tokens:
#     for token2 in tokens:
#         print(token1.text, token2.text, token1.similarity(token2))
#
# print(len(nlp.vocab.vectors))
# print(nlp.vocab.vectors.shape)
#
# tokens = nlp(u'dog cat gimbile')
# for token in tokens:
#     print(token.text, token.has_vector, token.vector_norm, token.is_oov)
#
# from scipy import spatial
# cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)
#
# king = nlp.vocab['king'].vector
# man = nlp.vocab['man'].vector
# woman = nlp.vocab['woman'].vector
#
# new_vector = king - man + woman
# computed_similarities = []
# for word in nlp.vocab:
#     if word.has_vector:
#         if word.is_lower:
#             if word.is_alpha:
#                 similarity = cosine_similarity(new_vector, word.vector)
#                 computed_similarities.append((word, similarity))
#
# computed_similarities = sorted(computed_similarities, key=lambda item: item[1], reverse=True)
# print([w[0].text for w in computed_similarities[:10]])
#
#
# ## 6.4 Sentiment Analysis Overview
# ## Nothing to implement :)
#
#
# ## 6.5 Sentiment Analysis with NLTK
# import nltk
# nltk.download('vader_lexicon')
#
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# sid = SentimentIntensityAnalyzer()
#
# a = "This is a good movie"
# print(sid.polarity_scores(a))
#
# b = "This was the best, most awesome movie EVER MADE!!!"
# print(sid.polarity_scores(b))
#
# c = 'This was the WORST movie that has ever disgraced the screen.'
# print(sid.polarity_scores(c))
#
# import pandas as pd
# df = pd.read_csv('./files_for_practice/TextFiles/amazonreviews.tsv', sep='\t')
# print(df.head())
# print(df['label'].value_counts())
#
# df.dropna(inplace=True)
# blanks = []
# for index, label, review in df.itertuples():
#     if type(review) == str:
#         if review.isspace():
#             blanks.append(index)
# df.drop(blanks, inplace=True)
#
# print(df.iloc[0]['review'])
# print(sid.polarity_scores(df.iloc[0]['review']))
#
# df['scores'] = df['review'].apply(lambda review: sid.polarity_scores(review))
# print(df.head())
#
# df['compound'] = df['scores'].apply(lambda dic: dic['compound'])
# print(df.head())
#
# df['comp_score'] = df['compound'].apply(lambda score: 'pos' if score >= 0 else 'neg')
# print(df.head())
#
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# print(accuracy_score(df['label'], df['comp_score']))
# print(confusion_matrix(df['label'], df['comp_score']))
# print(classification_report(df['label'], df['comp_score']))
#
#
# ## 6.6 Sentiment Analysis Code Along Movie Review Project
# import numpy as np
# import pandas as pd
# df = pd.read_csv('./files_for_practice/TextFiles/moviereviews.tsv', sep='\t')
# print(df.head())
#
# df.dropna(inplace=True)
# blanks = []
# for index, label, review in df.itertuples():
#     if type(review) == str:
#         if review.isspace():
#             blanks.append(index)
# df.drop(blanks, inplace=True)
#
# print(df['label'].value_counts())
#
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# sid = SentimentIntensityAnalyzer()
#
# df['scores'] = df['review'].apply(lambda review: sid.polarity_scores(review))
# df['compound'] = df['scores'].apply(lambda dic:dic['compound'])
# df['comp_score'] = df['compound'].apply(lambda score: 'pos' if score >= 0 else 'neg')
#
# print(df.head())
#
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# print(accuracy_score(df['label'], df['comp_score']))
# print(confusion_matrix(df['label'], df['comp_score']))
# print(classification_report(df['label'], df['comp_score']))
#
#
# ## 6.7 Sentiment Analysis Project Assessment
# import spacy
# nlp = spacy.load('en_core_web_lg')
#
# tokens = nlp(u'tree fruit apple')
# for t in tokens:
#     if t.has_vector:
#         print(t.vector)
#         print(t.vector.shape)
#
# for t1 in tokens:
#     for t2 in tokens:
#         print(t1.text, t2.text, t1.similarity(t2))
#
#
# def vector_math(a, b, c):
#     from scipy import spatial
#     cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)
#
#     a_vec = nlp.vocab[a].vector
#     b_vec = nlp.vocab[b].vector
#     c_vec = nlp.vocab[c].vector
#     new_vec = a_vec - b_vec + c_vec
#
#     similarity_list = []
#     for vocab in nlp.vocab:
#         if vocab.has_vector:
#             if vocab.is_lower:
#                 if vocab.is_alpha:
#                     similarity_list.append((vocab.text, cosine_similarity(vocab.vector, new_vec)))
#     similarity_list = sorted(similarity_list, key=lambda item: item[1], reverse=True)
#     print([item[0] for item in similarity_list[:10]])
#
# vector_math('wolf','dog','cat')
#
#
# def review_rating(in_rev):
#     from nltk.sentiment import SentimentIntensityAnalyzer
#     sid = SentimentIntensityAnalyzer()
#     score = sid.polarity_scores(in_rev)
#     if score['compound'] == 0: print ('Neutral')
#     elif score['compound'] > 0: print ('Positive')
#     else: print('Negative')
#
# review_rating("It was normal")

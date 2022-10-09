# ## 7.1 Introduction to Topic Modeling Section
# ## Nothing to implement :)
#
#
# ## 7.2 Overview of Topic Modeling
# ## Nothing to implement :)
#
#
# ## 7.3 Latent Dirichlet Allocation Overview
# ## Nothing to implement :)
#
#
# ## 7.4 Latent Dirichlet Allocation with Python - Part One
# import pandas as pd
# npr = pd.read_csv('./files_for_practice/TextFiles/npr.csv')
# print(npr.head())
# print(len(npr))
# print(npr['Article'][4000])
#
# from sklearn.feature_extraction.text import CountVectorizer
# cv = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
# dtm = cv.fit_transform(npr['Article'])
# print(dtm.shape)
#
# from sklearn.decomposition import LatentDirichletAllocation
# LDA = LatentDirichletAllocation(n_components=7, random_state=42)
# LDA.fit(dtm)
#
#
# ## 7.5 Latent Dirichlet Allocation with Python - Part Two
# import pandas as pd
# npr = pd.read_csv('./files_for_practice/TextFiles/npr.csv')
# print(len(npr))
#
# from sklearn.feature_extraction.text import CountVectorizer
# cv = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
# dtm = cv.fit_transform(npr['Article'])
#
# from sklearn.decomposition import LatentDirichletAllocation
# LDA = LatentDirichletAllocation(n_components=7, random_state=42)
# LDA.fit(dtm)
#
# # Grab the vocabulary of words
# print(len(cv.get_feature_names()))
# print(cv.get_feature_names()[50000])
# print(dtm.shape)
#
# # Grab the topics
# print(len(LDA.components_))
# print(LDA.components_.shape)
#
# single_topic = LDA.components_[0]
# print(single_topic.argsort())
#
# print(single_topic.argsort()[-10:])
# top_ten_words = single_topic.argsort()[-10:]
# for index in top_ten_words:
#     print(cv.get_feature_names()[index])
#
# # Grab the highest probability words per topic
# for i, topic in enumerate(LDA.components_):
#     print(f"THE TOP 15 WORDS FOR TOPIC #{i}")
#     print([cv.get_feature_names()[index] for index in topic.argsort()[-10:]])
#
# # Grab the probability of each topic per article
# topic_results = LDA.transform(dtm)
# print(topic_results.shape)
# print(topic_results[0])
# print(topic_results[0].round(2))
# print(topic_results[0].argmax())
#
# npr['Topic'] = topic_results.argmax(axis=1)
# print(npr.head())
#
#
# ## 7.6 Non-negative Matrix Factorization Overview
# ## Nothing to implement :)
#
#
# ## 7.7 Non-negative Matrix Factorization with Python
# import pandas as pd
# npr = pd.read_csv('./files_for_practice/TextFiles/npr.csv')
#
# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
# dtm = tfidf.fit_transform(npr['Article'])
# print(dtm.shape)
# print(tfidf.get_feature_names()[2300])
#
# from sklearn.decomposition import NMF
# nmf_model = NMF(n_components=7, random_state=42)
# nmf_model.fit(dtm)
#
# for i, topic in enumerate(nmf_model.components_):
#     print(f"THE TOP 15 WORDS FOR TOPIC #{i}")
#     print([tfidf.get_feature_names()[index] for index in topic.argsort()[-10:]])
#     print('\n')
#
# topic_results = nmf_model.transform(dtm)
# print(topic_results[0])
# print(topic_results[0].argmax())
# print(topic_results.argmax(axis=1))
#
# npr['Topic'] = topic_results.argmax(axis=1)
# print(npr.head())
#
# topic_dict = {0:'health', 1:'election', 2:'legis', 3:'poli', 4:'international', 5:'music', 6:'edu'}
# npr['Topic_Label'] = npr['Topic'].map(topic_dict)
# print(npr.head())
#
#
# ## 7.8 Topic Modeling Project - Overview
# import pandas as pd
# df = pd.read_csv('./files_for_practice/TextFiles/quora_questions.csv')
# print(df.head())
#
# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
# dtm = tfidf.fit_transform(df['Question'])
# print(dtm.shape)
#
# from sklearn.decomposition import NMF
# nmf_model = NMF(n_components=20, random_state=42)
# nmf_model.fit(dtm)
#
# for i, topic in enumerate(nmf_model.components_):
#     print(f"THE TOP 15 WORDS FOR TOPIC #{i}")
#     print([tfidf.get_feature_names()[index] for index in topic.argsort()[-15:]])
#     print('\n')
#
# topic_results = nmf_model.transform(dtm)
# df['Label'] = topic_results.argmax(axis=1)
# print(df.head())

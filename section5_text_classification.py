# ## 5.1 Introduction to Text Classification
# ## Nothing to implement :)
#
#
# ## 5.2 Machine Learning Overview
# ## Nothing to implement :)
#
#
# ## 5.3 Classification Metrics
# ## Nothing to implement :)
#
#
# ## 5.4 Confusion Matrix
# ## Nothing to implement :)
#
#
# ## 5.5 Scikit-Learn Primer - How to Use SciKit-Learn
# ## Nothing to implement :)
#
#
# ## 5.6 Scikit-Learn Primer - Code Along Part One
# import numpy as np
# import pandas as pd
# df = pd.read_csv('./files_for_practice/TextFiles/smsspamcollection.tsv',sep='\t')
# print(df.head())
#
# print(df.isnull().sum())
# print(len(df))
#
# print(df['label'])
# print(df['label'].unique())
# print(df['label'].value_counts())
#
# # import matplotlib.pyplot as plt
# # plt.xscale('log')
# # bins = 1.15**(np.arange(0,50))
# # plt.hist(df[df['label'] == 'ham']['length'], bins=bins, alpha=0.8)
# # plt.hist(df[df['label'] == 'spam']['length'], bins=bins, alpha=0.8)
# # plt.legend(('ham', 'spam'))
# # plt.show()
#
# # plt.xscale('log')
# # bins = 1.5**(np.arange(0,15))
# # plt.hist(df[df['label'] == 'ham']['punct'], bins=bins, alpha=0.8)
# # plt.hist(df[df['label'] == 'spam']['punct'], bins=bins, alpha=0.8)
# # plt.legend(('ham', 'spam'))
# # plt.show()
#
# from sklearn.model_selection import train_test_split
# X = df[['length','punct']]
# y = df['label']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
#
# from sklearn.linear_model import LogisticRegression
# lr_model = LogisticRegression(solver='lbfgs')
# lr_model.fit(X_train, y_train)
#
#
# ## 5.7 Scikit-Learn Primer - Code Along Part Two
# import numpy as np
# import pandas as pd
# df = pd.read_csv('./files_for_practice/TextFiles/smsspamcollection.tsv',sep='\t')
#
# from sklearn.model_selection import train_test_split
# X = df[['length','punct']]
# y = df['label']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
#
# from sklearn.linear_model import LogisticRegression
# lr_model = LogisticRegression(solver='lbfgs')
# lr_model.fit(X_train, y_train)
#
# from sklearn import metrics
# predictions = lr_model.predict(X_test)
# print(predictions)
#
# confusion_metrics = metrics.confusion_matrix(y_test, predictions)
# print(confusion_metrics)
#
# confusion_metrics_df = pd.DataFrame(metrics.confusion_matrix(y_test,predictions), index=['ham','spam'], columns=['ham','spam'])
# print(confusion_metrics_df)
#
# classification_report = metrics.classification_report(y_test, predictions)
# print(classification_report)
#
# accuracy = metrics.accuracy_score(y_test, predictions)
# print(accuracy)
#
#
# from sklearn.naive_bayes import MultinomialNB
# nb_model = MultinomialNB()
# nb_model.fit(X_train, y_train)
# predictions = nb_model.predict(X_test)
#
# print(metrics.confusion_matrix(y_test, predictions))
# print(metrics.classification_report(y_test, predictions))
#
#
# from sklearn.svm import SVC
# svc_model = SVC(gamma='auto')
# svc_model.fit(X_train,y_train)
# predictions = svc_model.predict(X_test)
#
# print(metrics.confusion_matrix(y_test,predictions))
# print(metrics.classification_report(y_test,predictions))
#
#
# ## 5.8 Text Feature Extraction Overview
# ## Nothing to implement :)
#
#
# ## 5.9 Text Feature Extraction - Code Along Implementations
# import numpy as np
# import pandas as pd
#
# df = pd.read_csv('./files_for_practice/TextFiles/smsspamcollection.tsv',sep='\t')
# print(df.head())
# print(df.isnull().sum())
#
# print(df['label'].value_counts())
#
#
# from sklearn.model_selection import train_test_split
# X = df['message']
# y = df['label']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#
# from sklearn.feature_extraction.text import CountVectorizer
# count_vect = CountVectorizer()
# count_vect.fit(X_train)
# X_train_counts = count_vect.transform(X_train)
# # WE CAN DO THE TWO ABOVE STEPS IN ONE STEP AS FOLLOWING:
# # X_train_counts = count_vect.fit_transform(X_train)
#
# print(X_train_counts)
# print(X_train.shape)
# print(X_train_counts.shape)
#
#
# ## 5.10 Text Feature Extraction - Code Along - Part Two
# import numpy as np
# import pandas as pd
#
# df = pd.read_csv('./files_for_practice/TextFiles/smsspamcollection.tsv',sep='\t')
#
# from sklearn.model_selection import train_test_split
# X = df['message']
# y = df['label']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#
# from sklearn.feature_extraction.text import CountVectorizer
# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(X_train)
#
# from sklearn.feature_extraction.text import TfidfTransformer
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# # # INSTEAD OF USING BOTH CountVectorizer AND TfidfTransformer WE CAN USE ONE SINGLE CLASS AS FOLLOWING:
# # # from sklearn.feature_extraction.text import TfidfVectorizer
# # # vectorizer = TfidfVectorizer()
# # # X_train_tfidf = vectorizer.fit_transform(X_train) # remember to use the original X_train set
#
# print(X_train_tfidf.shape)
# print(X_train_tfidf)
#
# # # _________________________________
# # #IN THE FOLLOWING WE TRY A CLASSIFICATION:
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # vectorizer = TfidfVectorizer()
# # X_tfidf = vectorizer.fit_transform(X)
# # X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.33, random_state=42)
# #
# # from sklearn.svm import LinearSVC
# # clf = LinearSVC()
# # clf.fit(X_train_tfidf, y_train)
# # predictions = clf.predict(X_test_tfidf)
# #
# # from sklearn.metrics import confusion_matrix, classification_report
# # print(confusion_matrix(y_test, predictions))
# # print(classification_report(y_test, predictions))
#
# # # INSTEAD OF ALL STEPS MENTIONED IN THE ABOVE CLASSIFIER WE CAN DEFINE A PIPELINE AS FOLLOWING:
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC
# from sklearn.pipeline import Pipeline
# text_clf = Pipeline([('tfidf', TfidfVectorizer()),('clf', LinearSVC())])
# text_clf.fit(X_train, y_train)
# predictions = text_clf.predict(X_test)
#
# from sklearn.metrics import confusion_matrix, classification_report
# print(confusion_matrix(y_test, predictions))
# print(classification_report(y_test, predictions))
#
# print(text_clf.predict(["Hi Jack, what time can I see you today?"]))
# print(text_clf.predict(["Congratulation, You have been selected as a winner. Call 44255 for free entry to contest"]))
#
#
# ## 5.11 Text Classification Code Along Project
# import numpy as np
# import pandas as pd
#
# df = pd.read_csv('./files_for_practice/TextFiles/moviereviews.tsv', sep='\t')
# print(df.head())
# print(len(df))
# print(df['review'][0])
#
# print(df.isnull().sum())
# df.dropna(inplace=True)
# print(df.isnull().sum())
# print(len(df))
#
# blanks = []
# for index, label, review in df.itertuples():
#     if review.isspace():
#         blanks.append(index)
# # print(blanks)
# df.drop(blanks, inplace=True)
# print(len(df))
#
# from sklearn.model_selection import train_test_split
# X = df['review']
# y = df['label']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC
#
# text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])
# text_clf.fit(X_train, y_train)
# predictions = text_clf.predict(X_test)
#
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# print(confusion_matrix(y_test, predictions))
# print(classification_report(y_test, predictions))
# print(accuracy_score(y_test, predictions))
#
#
# ## 5.12 Text Classification Assessment Overview
# import numpy as np
# import pandas as pd
#
# df = pd.read_csv('./files_for_practice/TextFiles/moviereviews2.tsv', sep='\t')
# print(df.head())
# print(len(df))
#
# print(df.isnull().sum())
# df.dropna(inplace=True)
# print(len(df))
#
# blanks = []
# for index, label, review in df.itertuples():
#     if review.isspace():
#         blanks.append(index)
# print(blanks)
# df.drop(blanks, inplace=True)
#
# print(df['label'].value_counts())
#
# from sklearn.model_selection import train_test_split
# X = df['review']
# y = df['label']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC
#
# text_clf = Pipeline([('tfidf', TfidfVectorizer()),('clf', LinearSVC())])
# text_clf.fit(X_train, y_train)
# predictions = text_clf.predict(X_test)
#
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# print(confusion_matrix(y_test, predictions))
# print(classification_report(y_test, predictions))
# print(accuracy_score(y_test, predictions))

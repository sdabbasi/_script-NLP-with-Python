# ## 4.1 Introduction to Section on POS and NER
# ## Nothing to say :)
#
#
# ## 4.2 Part of Speech Tagging
# import spacy
#
# nlp = spacy.load('en_core_web_sm')
#
# doc = nlp(u"The quick brown fox jumped over the lazy dog's back.")
#
# # for token in doc:
# #     print(f'{token.text:{10}} {token.pos_:{8}} {token.tag_:{6}} {spacy.explain(token.tag_)}')
#
# doc = nlp(u"I am reading books on NLP")
# word = doc[2]
# print(word)
# print(f'{word.text:{10}} {word.pos_:{8}} {word.tag_:{6}} {spacy.explain(word.tag_)}')
#
# doc = nlp(u"I read a book on NLP")
# word = doc[1]
# print(word)
# print(f"{word.text:{10}} {word.pos_:{8}} {word.tag_:{6}} {spacy.explain(word.tag_)}")
#
# doc = nlp(u"The quick brown fox jumped over the lazy dog's back.")
# POS_count = doc.count_by(spacy.attrs.POS)
# for k, v in sorted(POS_count.items(), key=lambda item: item[0]):
#     print(f'{k: <{3}}. {doc.vocab[k].text:{5}}: {v}')
# TAG_count = doc.count_by(spacy.attrs.TAG)
# for k, v in sorted(TAG_count.items(), key=lambda item: item[0]):
#     print(f'{k: <{25}}. {doc.vocab[k].text:{5}}: {v}')
# DEP_count = doc.count_by(spacy.attrs.DEP)
# for k, v in sorted(DEP_count.items(), key=lambda item: item[0]):
#     print(f'{k: <{20}}. {doc.vocab[k].text:{5}}: {v}')
#
#
# ## 4.3 Visualizing Part of Speech
# import spacy
# from spacy import displacy
#
# nlp = spacy.load('en_core_web_sm')
# doc = nlp(u"The quick brown fox jumped over the lazy dog's back.")
#
# for token in doc:
#     print(f'{token.text:{10}} {token.pos_:{7}} {token.dep_:{7}} {spacy.explain(token.dep_)}')
#
# # displacy.render(doc, style='dep', jupyter=True, options={'distance': 110})
# # displacy.serve(doc, style='dep', options={'distance': 110})
#
# # options = {'distance': 110, 'compact': True, 'color': 'yellow', 'bg': '#09a3d5', 'font': 'Times'}
# # displacy.serve(doc, style='dep', options=options)
#
# doc2 = nlp(u"This is a sentence. This is another, possibly longer sentence.")
# spans = list(doc2.sents)
# options = {'distance': 110, 'compact': True, 'color': 'yellow', 'bg': '#09a3d5', 'font': 'Times'}
# displacy.serve(spans, style='dep', options=options)
#
#
# ## 4.4 Named Entity Recognition  - Part One
# import spacy
#
# nlp = spacy.load('en_core_web_sm')
#
# def show_ents(in_doc):
#     if in_doc.ents:
#         for ent in in_doc.ents:
#             print(ent.text, ' - ', ent.label_, ' - ', str(spacy.explain(ent.label_)))
#     else:
#         print('No named entities found.')
#
#
# doc = nlp(u"May I go to Washington, DC next May to see the Washington Monument?")
# show_ents(doc)
#
# doc = nlp(u'Can I please borrow 500 dollars from you to buy some Microsoft stock?')
# show_ents(doc)
#
# doc = nlp(u'Tesla to build a U.K. factory for $6 million')
# show_ents(doc)
#
# from spacy.tokens import Span
#
# ORG = doc.vocab.strings[u'ORG']
# new_ent = Span(doc, 0, 1, label=ORG)
# doc.ents = list(doc.ents) + [new_ent]
# print(show_ents(doc))
#
#
# ## 4.5 Named Entity Recognition  - Part Two
# import spacy
#
# nlp = spacy.load('en_core_web_sm')
#
#
# def show_ents(in_doc):
#     if in_doc.ents:
#         for ent in in_doc.ents:
#             print(ent.text, ' - ', ent.label_, ' - ', str(spacy.explain(ent.label_)))
#     else:
#         print('No named entities found.')
#
#
# doc = nlp(u'Our company plans to introduce a new vacuum cleaner. '
#           u'If successful, the vacuum-cleaner will be our best product.')
# show_ents(doc)
#
# from spacy.matcher import PhraseMatcher
# matcher = PhraseMatcher(nlp.vocab)
#
# phrase_list = ['vacuum cleaner', 'vacuum-cleaner']
# phrase_patterns = [nlp(text) for text in phrase_list]
#
# matcher.add('newproduct', None, *phrase_patterns)
#
# found_matches = matcher(doc)
#
# from spacy.tokens import Span
#
# PROD = doc.vocab.strings[u"PRODUCT"]
# new_ents = [Span(doc, match[1], match[2], label=PROD) for match in found_matches]
#
# doc.ents = list(doc.ents) + [new_ents]
# print(show_ents(doc))
#
# doc = nlp(u'Originally priced at $29.50, the sweater was marked down to five dollars.')
# show_ents(doc)
# print(len([ent for ent in doc.ents if ent.label_=='MONEY']))


# ## 4.6 Visualizing Named Entity Recognition
# import spacy
# from spacy import displacy
#
# nlp = spacy.load('en_core_web_sm')
#
# doc = nlp(u'Over the last quarter Apple sold nearly 20 thousand iPods for a profit of $6 million. '
#          u'By contrast, Sony sold only 7 thousand Walkman music players.')
#
# displacy.render(doc, style='ent', jupyter=True)
# displacy.serve(doc, style='ent')
#
# for sent in doc.sents:
#     displacy.render(nlp(sent.text), style='ent', jupyter=True)
#
# options = {'ents': ['ORG', 'PRODUCT']}
# displacy.serve(doc, style='ent', options=options)
#
# colors = {'ORG': 'linear-gradient(90deg, #aa9cfc, #fc9ce7)', 'PRODUCT': 'radial-gradient(yellow, green)'}
# options = {'ents': ['ORG', 'PRODUCT'], 'colors':colors}
# displacy.serve(doc, style='ent', options=options)
#
#
# ## 4.7 Sentence Segmentation
# import spacy
# nlp = spacy.load('en_core_web_sm')
#
# doc = nlp(u'This is the first sentence. This is another sentence. This is the last sentence.')
# for sent in doc.sents:
#     print(sent)
#
# doc = nlp(u'"Management is doing great; leadership is doing right." -Peter Drucker')
# print(doc.text)
# for item in doc:
#     print(item)
# for sent in doc.sents:
#     print(sent)
#
# # We are going to define new segmentation rule and add to the pipeline
# def set_custom_boundaries(doc):
#     for token in doc[:-1]:
#         if token.text == ';':
#             doc[token.i+1].is_sent_start = True
#     return doc
#
# nlp.add_pipe(set_custom_boundaries, before='parser')
# print(nlp.pipe_names)
#
# doc2 = nlp(u'"Management is doing things right; leadership is doing the right things." -Peter Drucker')
# for sent in doc2.sents:
#     print(sent)
#
# We are going to replace the existing segmentation rule
# import spacy
# from spacy.pipeline import SentenceSegmenter
#
# nlp = spacy.load('en_core_web_sm')
#
#
# def split_on_newlines(doc):
#     start = 0
#     seen_newline = False
#     for word in doc:
#         if seen_newline:
#             yield doc[start:word.i]
#             start = word.i
#             seen_newline = False
#         elif word.text.startswith('\n'):  # handles multiple occurrences
#             seen_newline = True
#     yield doc[start:]  # handles the last group of tokens
#
#
# sbd = SentenceSegmenter(nlp.vocab, strategy=split_on_newlines)
# nlp.add_pipe(sbd)
# print(nlp.pipe_names)
#
# my_string = u"This is a sentence. This is another.\n\nThis is a \nthird sentence."
# doc = nlp(my_string)
#
# for sent in doc.sents:
#     print(sent)
#
#
# ## 4.8 Part Of Speech Assessment
# import spacy
# from spacy import displacy
#
# nlp = spacy.load('en_core_web_sm')
#
# with open('../my_project/files_for_practice/peterrabbit.txt') as f:
#     content = f.read()
# doc = nlp(content)
#
# for token in nlp(list(doc.sents)[2].text):
#     print(f'{token.text:{10}} {token.pos_:{10}} {token.tag_:{10}} {spacy.explain(token.tag_)}')
#
# POS_count = doc.count_by(spacy.attrs.POS)
# print(POS_count)
# for k, v in sorted(POS_count.items(), key=lambda item: item[0]):
#     print(f'{k: <{5}}. {doc.vocab[k].text:{5}}: {v}')
#
#
# container_all = 0
# container_noun = 0
# for k, v in POS_count.items():
#     if k == nlp.vocab.strings['NOUN']:
#         container_noun = v
#     container_all += v
# print(container_noun*100/container_all)
#
# displacy.serve(nlp(list(doc.sents)[2].text), style='dep',options={'distance': 110})
#
# for ent in doc.ents[:2]:
#     print(ent.text, ' - ', ent.label_, ' - ', str(spacy.explain(ent.label_)))
#
# list = []
# for sent in doc.sents:
#     if nlp(sent.text).ents:
#         list.append(sent)
# print(len(list))
#
# displacy.serve(nlp(list(doc.sents)[0].text), style='ent')

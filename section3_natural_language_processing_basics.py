# ## 3.1 Introduction to Natural Language Processing
# ## Nothing to say :)
#
#
# ## 3.2 Spacy Setup and Overview
# ## Nothing to say :)
#
#
# ## 3.3 What is Natural Language Processing?
# ## Nothing to say :)
#
#
# ## 3.4 Spacy Basics
# import spacy
#
# nlp = spacy.load('en_core_web_sm')
#
# doc = nlp(u'Tesla is looking at buying U.S. startup for $6 million')
# for token in doc:
#     print(token.text, token.pos, token.pos_, token.dep_)
#
# print(nlp.pipeline)
# print(nlp.pipe_names)
#
# doc2 = nlp(u"Tesla isn't     looking into startups anymore.")
# for token in doc2:
#     print(token.text, token.pos, token.pos_, token.dep_)
#
# doc3 = nlp(u'Although commonly attributed to John Lennon from his song "Beautiful Boy", \
#             the phrase "Life is what happens to us while we are making other plans" was written by \
#             cartoonist Allen Saunders and published in Reader\'s Digest in 1957, when Lennon was 17."')
# life_quote = doc3[16:30]
# print(life_quote)
# print(type(life_quote))
#
# doc4 = nlp(u"This is the first sentence. This is another sentence. This is the last sentence.")
# for sentence in doc4.sents:
#     print(sentence)
# print(doc4[6].is_sent_start)
# print(doc4[8].is_sent_start)
#
#
# ## 3.5 Tokenization - Part 1
# import spacy
#
# nlp = spacy.load('en_core_web_sm')
#
# mystring = '"We\'re moving to L.A.!"'
# doc = nlp(mystring)
# for token in doc:
#     print(token)
#
# doc2 = nlp(u"We're here to help! Send snail-mail, email support@oursite.com or visit us at http://www.oursite.com!")
# for token in doc2:
#     print(token)
#
# doc3 = nlp(u'A 5km NYC cab ride costs $10.30')
# for t in doc3:
#     print(t)
#
# doc4 = nlp(u"Let's visit St. Louis in the U.S. next year.")
# for t in doc4:
#     print(t)
#
# doc5 = nlp(u'It is better to give than to receive.')
# print(doc5[2])
# print(doc5[2:5])
# print(doc5[-4:])
#
# doc6 = nlp(u'Apple to build a Hong Kong factory for $6 million')
# for token in doc6:
#     print(token.text, end=' | ')
# print('\n----')
# for ent in doc6.ents:
#     print(ent.text+' - '+ent.label_+' - '+str(spacy.explain(ent.label_)))
#
# doc7 = nlp(u"Autonomous cars shift insurance liability toward manufacturers.")
# for chunk in doc7.noun_chunks:
#     print(chunk.text)
#
#
# ## 3.6 Tokenization - Part 2
# import spacy
# from spacy import displacy
#
# nlp = spacy.load('en_core_web_sm')
#
# doc = nlp(u'Apple is going to build a U.K. factory for $6 million.')
# displacy.render(doc, style='dep', jupyter=True, options={'distance': 110})
# displacy.serve(doc, style='dep', options={'distance': 110})
#
# doc2 = nlp(u'Over the last quarter Apple sold nearly 20 thousand iPods for a profit of $6 million.')
# displacy.render(doc2, style='ent', jupyter=True)
# displacy.serve(doc2, style='ent')
#
# doc3 = nlp(u'This is a sentence.')
# displacy.serve(doc3, style='dep')
#
#
# ## 3.7 Stemming
# from nltk.stem.porter import PorterStemmer
#
# p_stemmer = PorterStemmer()
# words = ['run','runner','running','ran','runs','easily','fairly', 'fairness']
# for word in words:
#     print(word+' --> '+p_stemmer.stem(word))
#
# from nltk.stem.snowball import SnowballStemmer
# s_stemmer = SnowballStemmer(language='english')
# for word in words:
#     print(word+' --> '+s_stemmer.stem(word))
#
# words_2 = ['generous', 'generation', 'generously', 'generate']
# for word in words_2:
#     print(word+' --> '+p_stemmer.stem(word))
# for word in words_2:
#     print(word+' --> '+s_stemmer.stem(word))
#
#
# ## 3.8 Lemmatization
# import spacy
#
# nlp = spacy.load('en_core_web_sm')
#
# doc1 = nlp(u"I am a runner running in a race because I love to run since I ran today")
# for token in doc1:
#     print(token.text, '\t', token.pos_, '\t', token.lemma, '\t', token.lemma_)
#
# def show_lemmas(text):
#     for token in text:
#         print(f'{token.text:{10}} {token.pos_:{10}} {token.lemma:<{25}} {token.lemma_:{10}}')
#
# show_lemmas(doc1)
#
# doc2 = nlp(u"I saw eighteen mice today!")
# show_lemmas(doc2)
#
#
# ## 3.9 Stop Words
# import spacy
#
# nlp = spacy.load('en_core_web_sm')
#
# print(nlp.Defaults.stop_words)
# print(len(nlp.Defaults.stop_words))
#
# print(nlp.vocab['myself'].is_stop)
# print(nlp.vocab['mystery'].is_stop)

# nlp.Defaults.stop_words.add("btw")
# nlp.vocab['btw'].is_stop = True
# print(nlp.vocab['btw'].is_stop)
# print(len(nlp.Defaults.stop_words))
#
# nlp.Defaults.stop_words.remove('beyond')
# nlp.vocab['beyond'].is_stop = False
# print(nlp.vocab['beyond'].is_stop)
# print(len(nlp.Defaults.stop_words))
#
#
# ## 3.10 Phrase Matching and Vocabulary - Part One
# import spacy
# from spacy.matcher import Matcher
#
# nlp = spacy.load('en_core_web_sm')
# matcher = Matcher(nlp.vocab)
# doc = nlp(u"The Solar Power industry continues to grow as demand \
#             for solarpower increases. Solar-power cars are gaining popularity.")
#
# # SolarPower
# pattern1 = [{'LOWER': 'solarpower'}]
# # solar-power
# pattern2 = [{'LOWER': 'solar'}, {'IS_PUNCT': True}, {'LOWER': 'power'}]
# # solar power
# pattern3 = [{'LOWER': 'solar'}, {'LOWER': 'power'}]
#
# matcher.add('Solarpower', None, pattern1, pattern2, pattern3)
# found_matches = matcher(doc)
# print(found_matches)
# for match_id, start, end in found_matches:
#     string_id = nlp.vocab.strings[match_id]  # get string representation
#     span = doc[start:end]                    # get the matched span
#     print(match_id, string_id, start, end, span.text)
#
# matcher.remove('Solarpower')
#
# # SolarPower
# pattern1 = [{'LOWER': 'solarpower'}]
# # solar...power
# pattern2 = [{'LOWER': 'solar'}, {'IS_PUNCT': True, 'OP':'*'}, {'LOWER': 'power'}]
#
# matcher.add('Solarpower', None, pattern1, pattern2)
# found_matches = matcher(doc)
# for match_id, start, end in found_matches:
#     string_id = nlp.vocab.strings[match_id]  # get string representation
#     span = doc[start:end]                    # get the matched span
#     print(match_id, string_id, start, end, span.text)
#
# doc2 = nlp(u"The Solar--Power is solarpower yay!")
# found_matches = matcher(doc2)
# for match_id, start, end in found_matches:
#     string_id = nlp.vocab.strings[match_id]  # get string representation
#     span = doc2[start:end]                    # get the matched span
#     print(match_id, string_id, start, end, span.text)
#
# doc3 = nlp(u'Solar-powered energy runs cars.')
# pattern1 = [{'LOWER': 'solar'}, {'IS_PUNCT': True, 'OP':'*'}, {'LEMMA': 'power'}]
# matcher.add('Solarpower', None, pattern1)
# found_matches = matcher(doc3)
# for match_id, start, end in found_matches:
#     string_id = nlp.vocab.strings[match_id]  # get string representation
#     span = doc3[start:end]                    # get the matched span
#     print(match_id, string_id, start, end, span.text)
#
#
# ## 3.11 Phrase Matching and Vocabulary - Part Two
# import spacy
# from spacy.matcher import PhraseMatcher
#
# nlp = spacy.load('en_core_web_sm')
# matcher = PhraseMatcher(nlp.vocab)
#
# with open('./files_for_practice/reaganomics.txt') as f:
#     doc3 = nlp(f.read())
#
# phrase_list = ['voodoo economics', 'supply-side economics', 'trickle-down economics', 'free-market economics']
# phrase_patterns = [nlp(text) for text in phrase_list]
#
# print(type(phrase_patterns[0]))
# print(phrase_patterns)
# print(*phrase_patterns)
#
# matcher.add('EconMatcher', None, *phrase_patterns)
# found_matches = matcher(doc3)
# for match_id, start, end in found_matches:
#     string_id = nlp.vocab.strings[match_id]  # get string representation
#     span = doc3[start:end]                    # get the matched span
#     print(match_id, string_id, start, end, span.text)
#
#
# ## 3.12 NLP Basics Assessment Overview
# import spacy
# from spacy.matcher import Matcher
# nlp = spacy.load('en_core_web_sm')
#
# with open('../my_project/files_for_practice/owlcreek.txt', 'r') as f:
#     doc = nlp(f.read())
#
# print(doc[:36])
# print(len(doc))
# print(len([sent for sent in doc.sents]))
# print([sent for sent in doc.sents][2])
#
# sentences = [sent for sent in doc.sents]
# for token in sentences[2]:
#     print(f"{token.text:{10}} {token.pos_:{10}} {token.dep_:{10}} {token.lemma_:{10}}")
#
# matcher = Matcher(nlp.vocab)
# pattern1 = [{'LOWER': 'swimming'}, {'IS_SPACE': True, 'OP': '*'}, {'LOWER': 'vigorously'}]
# matcher.add("Swimming", None, pattern1)
# found_matches = matcher(doc)
#
# for match_id, start, end in found_matches:
#     string_id = nlp.vocab.strings[match_id]  # get string representation
#     span = doc[start:end]                    # get the matched span
#     print(match_id, string_id, start, end, span.text)
#
# sentences = [sent for sent in doc.sents]
# for match_id, start, end in found_matches:
#     for sent in sentences:
#         if end <= sent.end:
#             span = doc[start:end]
#             print(sent)
#             break
#
#
# ## 3.13 NLP Basics Assessment Solution
# Nothing more to say

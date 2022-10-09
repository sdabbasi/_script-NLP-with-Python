# ## 2.1 Introduction to Python Text Basics
# ## Nothing to say :)
#
#
# ## 2.2 Working with Text Files with Python - Part One
# person = "Jose"
# print("my name is {}".format(person))
# print(f"my name is {person}")
#
# d = {'a': 123, 'b': 456}
# print(f"my number is {d['a']}")
#
# mylist = [0,1,2]
# print(f"my number is {mylist[0]}")
#
# library = [('Author', 'Topic', 'Pages'),
#            ('Twain', 'Rafting in water alone', 601),
#            ('Feynman', 'Physics', 95),
#            ('Hamilton', 'Mythology', 144)]
# for author, topic, pages in library:
#     print(f"{author:{10}} {topic:{30}} {pages:{10}}")
#
# for author, topic, pages in library:
#     print(f"{author:{10}} {topic:{30}} {pages:>{10}}")
#
# for author, topic, pages in library:
#     print(f"{author:{10}} {topic:{30}} {pages:->{10}}")
#
# for author, topic, pages in library:
#     print(f"{author:{10}} {topic:{30}} {pages:*>{10}}")
#
#
# from datetime import datetime
# today = datetime(year=2020, month=2, day=28)
# print(today)
# print(f"{today}")
# print(f"{today:%B %d; %Y}")         # check out for datetime formatting https://strftime.org/
#
#
# ## 2.3 Working with Text Files with Python - Part Two
# myfile = open('./files_for_practice/test.txt')
# print(myfile.read())
# print(myfile.read())
#
# myfile.seek(0)
# content = myfile.read()
# print(content)
# myfile.close()
# print("----------------")
#
# myfile = open('./files_for_practice/test.txt')
# print(myfile.readlines())
#
# myfile.seek(0)
# mylines = myfile.readlines()
# for line in mylines:
#     print(line.split()[0])
# myfile.close()
# print("----------")
#
# myfile = open('./files_for_practice/test2.txt', mode='w+')    # this mode overwrite the file (the previous text would be deleted)
# print(myfile.read())
# myfile.write("My brand new text")
# myfile.seek(0)
# print(myfile.read())
# myfile.close()
# print("---------------")
#
# myfile = open('./files_for_practice/test2.txt', mode='a+')
# myfile.write('\nMy first line in a+ mode opening')
# myfile.close()
# myfile = open('./files_for_practice/test2.txt')
# print(myfile.read())
# myfile.close()
# print("---------")
#
# myfile = open('./files_for_practice/test2.txt', mode='a+')
# myfile.write('\nThis is an added line because I am sing a+ mode')
# myfile.seek(0)
# print(myfile.read())
# myfile.close()
#
# with open('./files_for_practice/test2.txt', 'r') as myNewFile:
#     myvaribale = myNewFile.readlines()
# print(myvaribale)
# print('------------')
#
#
# ## 2.4 Working with PDFs
# import PyPDF2
# my_file = open('./files_for_practice/US_Declaration.pdf', mode='rb')     # use rb mode for pdf
# pdf_reader = PyPDF2.PdfFileReader(my_file)
# print(pdf_reader.numPages)
#
# page_one = pdf_reader.getPage(0)
# myText = page_one.extractText()
# print(myText)
# myfile.close()
# print('-----------')
#
# f = open('./files_for_practice/US_Declaration.pdf', mode='rb')
# pdf_reader = PyPDF2.PdfFileReader(f)
# first_page = pdf_reader.getPage(0)
# pdf_writer = PyPDF2.PdfFileWriter()
# pdf_writer.addPage(first_page)
# pdf_output = open('./files_for_practice/US_Declaration.pdf', mode='wb')
# pdf_writer.write(pdf_output)
# pdf_output.close()
# f.close()
#
# brand_new = open('./files_for_practice/my_brand_new.pdf', mode='rb')
# pdf_reader = PyPDF2.PdfFileReader(brand_new)
# print(pdf_reader.numPages)
# print(pdf_reader.getPage(0).extractText())
# print('----------')
#
# f = open('./files_for_practice/US_Declaration.pdf', 'rb')
# pdf_text = []
# pdf_reader = PyPDF2.PdfFileReader(f)
# for page in range(pdf_reader.numPages):
#     page_content = pdf_reader.getPage(page)
#     pdf_text.append(page_content.extractText())
# f.close()
# print(len(pdf_text))
# for page in pdf_text:
#     print(page)
#     print('\n\n\n\n\n\n\n')
#
#
# ## 2.5 Regular Expressions Part One
# text = "The phone number of the agent is 408-555-1234. Call soon!"
# print("408-555-1234" in text)
#
# import re
# pattern = "phone"
# my_match = re.search(pattern, text)
# print(my_match.span())
# print(my_match.start())
# print(my_match.end())
#
# text = "My phone is a new phone"
# match = re.search("phone", text)
# print(match.span())
# print("----------------------")
#
# all_matches = re.findall("phone", text)
# print(len(all_matches))
# for match in re.finditer("phone", text):
#     print(match.span())
# print("-------")
#
# text = "My phone number is 111-222-3333"
# pattern = r"\d\d\d-\d\d\d-\d\d\d\d"
# phone = re.search(pattern, text)
# print(phone.group())
# pattern = r"\d{3}-\d{3}-\d{4}"
# phone = re.search(pattern, text)
# print(phone.group())
# print("--------------")
#
#
# ## 2.6 Regular Expressions Part Two
# text = "My phone number is 111-222-3333"
# pattern = r"(\d{3})-(\d{3})-(\d{4})"
# phone = re.search(pattern, text)
# print(phone.group())
# print(phone.group(3))
# print("-------------")
#
# result = re.search(r"man|woman", "This man was here")
# print(result.group())
# result = re.search(r"man|woman", "This woman was here")
# print(result.group())
#
# result = re.findall(r".at", "The cat in the hat sat splat")
# print(result)
# result = re.findall(r"..at", "The cat in the hat sat splat")
# print(result)
# print("------")
#
# result = re.findall(r"\d$", "This ends with a number 2")
# print(result)
# result = re.findall(r"^\d", "1 is the loneliest number")
# print(result)
# print("-------")
#
# phrase = "there are 3 numbers 34 inside 367 this sentence"
# result = re.findall(r"[^\d]", phrase)
# print(result)
# result = re.findall(r"[^\d]+", phrase)
# print(result)
# print("-------")
#
# test_phrase = "This is a string! but it has punctuation. How to remove it?"
# result = re.findall(r"[^!.? ]+", test_phrase)
# print(result)
# box = ' '.join(result)
# print(box)
# print("-------")
#
# text = "Only find the hyphen-words. Were are the long-ish dash words?"
# result = re.findall(r"[\w]+-[\w]+", text)
# print(result)
#
#
# ## 2.7 Python Text Basics - Assessment Overview
# abbr = 'NLP'
# full_text = 'Natural Language Processing'
# print(f"{abbr} stands for {full_text}")
#
# myfile = open('./files_for_practice/contacts.txt', mode='r')
# fields = myfile.readline()
# print(fields)
# myfile.close()
#
# import PyPDF2 as pdf
# myfile = open('./files_for_practice/Business_Proposal.pdf', mode='rb')
# my_reader = pdf.PdfFileReader(myfile)
# page_two_text = my_reader.getPage(1).extractText()
# print(page_two_text)
# myfile.close()
#
# with open('contacts.txt', mode='a+') as myContact:
#     with open('./files_for_practice/Business_Proposal.pdf', mode='rb') as myPdf:
#         my_reader = pdf.PdfFileReader(myPdf)
#         page_two_text = my_reader.getPage(1).extractText()
#         splited_page_two_text = page_two_text.split(maxsplit=1)[1]
#         myContact.write(splited_page_two_text)
#
# import re
# with open('./files_for_practice/Business_Proposal.pdf', mode='rb') as myPdf:
#     my_reader = pdf.PdfFileReader(myPdf)
#     page_two_text = my_reader.getPage(1).extractText()
#
#     pattern = r"[\w]+@[\w]+"
#     print(re.findall(pattern, page_two_text))
#
#
# ## 2.8 Python Text Basics - Assessment Solutions
# abbr = 'NLP'
# full_text = 'Natural Language Processing'
# # Enter your code here:
# print(f'{abbr} stands for {full_text}')
#
# with open('contacts.txt') as c:
#     fields = c.read()
# print(fields)
#
# import PyPDF2
# # Open the file as a binary object
# f = open('./files_for_practice/Business_Proposal.pdf','rb')
# # Use PyPDF2 to read the text of the file
# pdf_reader = PyPDF2.PdfFileReader(f)
# # Get the text from page 2 (CHALLENGE: Do this in one step!)
# page_two_text = pdf_reader.getPage(1).extractText()
# # Close the file
# f.close()
# # Print the contents of page_two_text
# print(page_two_text)
#
# with open('./files_for_practice/contacts.txt','a+') as c:    # Simple Solution:
#     c.write(page_two_text)
#     c.seek(0)
#     print(c.read())
#
# with open('./files_for_practice/contacts.txt','a+') as c:    # CHALLENGE Solution
#     c.write(page_two_text[8:])
#     c.seek(0)
#     print(c.read())
#
# import re
# # Enter your regex pattern here. This may take several tries!
# pattern = r'\w+@\w+.\w{2,}'
# print(re.findall(pattern, page_two_text))


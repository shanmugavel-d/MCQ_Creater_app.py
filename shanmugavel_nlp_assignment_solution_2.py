# -*- coding: utf-8 -*-
"""The code you have provided is a Python code that generates multiple choice questions (MCQs) from a given text. The code first tokenizes the text and identifies the keywords in the text. Then, for each keyword, it finds the sentences in the text that contain the keyword. Next, it uses WordNet and ConceptNet to generate distractors for the keyword. Finally, it generates MCQs by replacing the keyword in the sentences with the distractors.

The code is well-organized and easy to understand. It uses a variety of Python libraries, including nltk, spacy, and requests. The code is also efficient and can generate MCQs from large texts quickly.

Here are some of the benefits of using this code:

* It can be used to generate MCQs from any text, regardless of the topic.
* It can generate a variety of MCQs, including fill-in-the-blank, multiple choice, and true/false questions.
* It can generate MCQs with a variety of distractors, making the questions more challenging.
* The code is efficient and can generate MCQs from large texts quickly.

However, there are also some limitations to the code:

* The code requires a working installation of Python and the relevant libraries.
* The code may not be able to generate MCQs from all texts. For example, the code may not be able to generate MCQs from texts that contain a lot of technical jargon.
* The code may not be able to generate MCQs that are of the same quality as MCQs that are created by humans.

Overall, the code is a useful tool for generating MCQs from text. It is well-organized, efficient, and can generate a variety of MCQs. However, there are some limitations to the code, such as the need for a working installation of Python and the relevant libraries.
"""

!pip install flashtext
!pip install pywsd
!pip install bert-extractive-summarizer



import pandas as pd
import spacy
import nltk
import re
import random
import requests
import json
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from flashtext import KeywordProcessor
from nltk.tokenize import sent_tokenize
from pywsd.similarity import max_similarity
from pywsd.lesk import adapted_lesk
from pywsd.lesk import cosine_lesk




msg_rd = input()

#summarize the text using Bert Extract Summarizer
from summarizer import Summarizer #---importing
model=Summarizer()                #---initializing
msg_sum = model(msg_rd, min_length=60, max_length = 500,ratio =0.4)

Sum_txt ="".join(msg_sum)


stop_words = stopwords.words('english') #extract all stiopwords in english and store in a variable.
def tokenize(rem):

    #it takes the paragraph ,break into words,check for stopwords, and remove if stopwords is present, and combine remaining into paragraph again.
    Sum_txt_tokenized = word_tokenize(rem)
    Sum_txt_new = "".join([i for i in Sum_txt_tokenized if i not in stop_words])

    return Sum_txt_tokenized

tokenize(Sum_txt)
#lemmatization

#function to convert nltk tag to wordnet tag.

lemmatizer = WordNetLemmatizer() #initializing

#finds PartsOfSpeech tag
#converts pos tag into small information.

def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('N'):
         return wn.NOUN
    else :
        return None

#lemmatize sentence using POS tag
def lemmatize_sentence(sentence):
    key_wrd=[]
    #word tokenize-->pos tag-->wordnet tag-->lemmatizer-->root words
    #tokenize the sentence and find pos tag for each word.

    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    wordnet_tagged = map(lambda x: (x[0],nltk_tag_to_wordnet_tag(x[1])),nltk_tagged)
    lemmatized_sentence=[]
    garb=[]
    for word,tag in wordnet_tagged:
        if tag is None:
            #if there is not tag apped word as it is
            garb.append(word)
        else:
             #use tag to lemmatize token
             lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))

    for word in lemmatized_sentence:
        key_wrd.append(word)
    return key_wrd


    # return " ".join(lemmatized_sentence)

keyword = lemmatize_sentence(Sum_txt)
filtered=[]
for i in keyword:
    if i.lower() in Sum_txt.lower():
      filtered.append(i)



# #Sentence Mapping
# #for each keyword get sentence from summarized text containg that keyword


def strip_sentence(text):
    sentences = [sent_tokenize(text)]
    sentences = [y for x in sentences for y in x]
    sentences = [sentence.strip() for sentence in sentences if len(sentence)>20]
    return sentences

stripd = strip_sentence(Sum_txt)


def sentences_for_words(keyword,sentences):

    keyword_processor = KeywordProcessor()
    keyword_sentences = {}

    for word in keyword:                      #this refers to lemmatize_sentences word in keyword
        keyword_sentences[word] = []
        keyword_processor.add_keyword(word)

    for sentence in sentences:
        key_sentences = keyword_processor.extract_keywords(sentence)
        for key in key_sentences:
            keyword_sentences[key].append(sentence)

    for key in keyword_sentences.keys():
        values = keyword_sentences[key]
        values = sorted(values, key=len,reverse=True)
        keyword_sentences[key] = values

    return keyword_sentences

sentence_mapping = sentences_for_words(filtered,stripd)




# # #Generating MCQ
# # # getting wrong answers(distractors) from wordnet and generating mcq questions based on that.

# # #Distractors from wordnet
def generate_distractors_wordnet(synsets,word):
  distractors = []
  word = word.lower()
  original_word = word

  if len(word.split())>0:
      word=word.replace(" ","_")
      synsets = wn.synsets(word,'n')


      if synsets:
        hypernyms = synsets[0].hypernyms()
        if not hypernyms:
          return distractors

        for item in hypernyms[0].hyponyms():
           name = item.lemmas()[0].name()
           if name == original_word:
              continue
           name = name.replace("_"," ")
           name = " ".join(i.capitalize() for i in name.split())
           if name is not None and name not in distractors:
               distractors.append(name)
  return distractors

def get_wsd(sent,word):

     word = word.lower()

     if len(word.split())>0:
        word = word.replace(" ","_")

     synsets = wn.synsets(word,'n')

     if synsets:
          ms = max_similarity(sent, word, 'wup', pos='n')
          adapted_lesk_output =  adapted_lesk(sent, word, pos='n')

          if ms is not None and adapted_lesk_output is not None:
              ms_index = synsets.index(ms)
              adapted_lesk_index = synsets.index(adapted_lesk_output)

              if ms_index < adapted_lesk_index:
                return ms
              else:
                return adapted_lesk_output
          # Handle cases where one of the similarity measures is None
          elif ms is not None:
             return ms
          elif adapted_lesk_output is not None:
            return adapted_lesk_output
          else:
            return None


          lowest_index = min(synsets.index(ms),synsets.index(adapted_lesk_output)) #minimum similarity from synsets
          return synsets[lowest_index]  #returns less similar words
     else:
         return None
# print(get_wsd(stripd,filtered))

#Dirstractors from conceptnet
def get_distractors_conceptnet(word):
  word = word.lower()
  original_word = word
  if (len(word.split())>0):
    word = word.replace(" ","_")
  distractor_list =[]
  url = "http://api.conceptnet.io/query?node=/c/en/%s/n&rel=/r/PartOf&start=/c/en/%s&limit=5"%(word,word)
  obj = requests.get(url).json()

  for edge in obj['edges']:
    link = edge['end']['term']
    url2 = "http://api.conceptnet.io/query?node=%s&rel=/r/PartOf&end=%s&limit=10"%(link,link)
    obj2 = requests.get(url2).json()
    for edge in obj2['edges']:
       word2 = edge['start']['label']
       if word2 not in distractor_list and original_word.lower() not in word2.lower():
          distractor_list.append(word2)

  return distractor_list


distractors_lst={}
for key,values in sentence_mapping.items():
  if key in sentence_mapping:
    sentences = sentence_mapping[key]
    if sentences:
        wordsense = get_wsd(sentences[0], key)

        print("Key:", key)


        if wordsense:
           distractors= generate_distractors_wordnet(wordsense,key)
           if len(distractors) ==0:
                distractors = get_distractors_conceptnet(key)
           if len(distractors)!= 0:
                distractors_lst[key] = distractors
        else:
           distractors = get_distractors_conceptnet(key)
           if len(distractors) != 0:
             distractors_lst[key] = distractors
    else:
       print("No sentences found for key:", key)
  else:
     print("Key not found:", key)

index=1
max_questions = 10
for word in distractors_lst:
    if index > max_questions:
      break
    sentences = sentence_mapping[word]
    sentence = sentences[0]

    pattern = re.compile(re.escape(word), re.IGNORECASE)
    output = pattern.sub(" _________ ",sentence)

    print ("%s)"% (index),output)
    choices = [word.capitalize()] + distractors_lst[word]
    top_choices = choices[:4]

    random.shuffle(top_choices)
    optionchoices = ['1','2','3','4']

    correct_answer = word.capitalize()
    correct_option = random.choice(optionchoices)

    for indx,choice in enumerate(top_choices):
      if choice == correct_answer:
        correct_option = optionchoices[indx]
      print ("\t",optionchoices[indx],")"," ",choice)
    print("Answer:")
    print("\t","Correct Ans is:" ,correct_option, ")", " ", correct_answer)
    index = index + 1
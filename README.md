# nlp-scam

## parse tree
```py
import nltk
grammar = nltk.CFG.fromstring("""
S -> NP VP
NP -> Det Nom
VP -> V NP
Nom -> Adj Nom | N
Det -> 'the'
Adj -> 'little' | 'angry' | 'frightened'
N -> 'squirrel' | 'bear'
V -> 'chased' """)
sentence = 'the angry bear chased the frightened little squirrel'.split() 
top_down=nltk.parse.TopDownChartParser(grammar)
left_right=nltk.parse.LeftCornerChartParser(grammar)
depth_first=nltk.parse.BottomUpLeftCornerChartParser(grammar)
for t in top_down.parse(sentence):
  print(t)
# parse(sentence).draw()
```
## edit distance
```py
def edit_distance(str1, str2, a, b):
    string_matrix = [[0 for i in range(b+1)] for i in range(a+1)]

    for i in range(a+1):
        for j in range(b+1):

            if i == 0:
                string_matrix[i][j] = j 
            elif j == 0:
                string_matrix[i][j] = i   
            elif str1[i-1] == str2[j-1]:
                string_matrix[i][j] = string_matrix[i-1][j-1]  
            else:
                string_matrix[i][j] = 1 + min(string_matrix[i][j-1],      
                                       string_matrix[i-1][j],      
                                       string_matrix[i-1][j-1])    

    return string_matrix[a][b]

str1 = 'Saturday'
str2 = 'Sunday'

print('No. of Operations required :',edit_distance(str1, str2, len(str1), len(str2)))
```
## n-gram
```py
from nltk.util import ngrams
from nltk.probability import FreqDist
from nltk.probability import ConditionalFreqDist
corpus = [
    "<s> I am Henry </s>",
    "<s> I like college </s>",
    "<s> Do Henry like college </s>",
    "<s> Henry I am </s>",
    "<s> Do I like Henry </s>",
    "<s> Do I like college</s>",
    "<s> I do like Henry</s>"
]
tokenized_corpus = [sentence.split() for sentence in corpus]
trigrams = []
for sentence in tokenized_corpus:
    trigrams.extend(ngrams(sentence, 3, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
trigram_freqdist = FreqDist(trigrams)
cfd = ConditionalFreqDist()
for trigram in trigram_freqdist:
    condition = (trigram[0], trigram[1])
    word = trigram[2]
    cfd[condition][word] += trigram_freqdist[trigram]
sentence="Do I like"
word=sentence.split()
condition = (word[-2],word[-1])
predicted_word = cfd[condition].max()

print(predicted_word)
```
## text summariation
```py
import nltk 
nltk.download('punkt') 
from sumy.parsers.plaintext import PlaintextParser 
from sumy.nlp.tokenizers import Tokenizer 
from sumy.summarizers.lsa import LsaSummarizer 
from sumy.utils import get_stop_words
print('ENTER YOUR TEXT HERE : ') 
text = input()
parser = PlaintextParser.from_string(text, Tokenizer("english"))
summarizer = LsaSummarizer()
summarizer.stop_words = get_stop_words("english")
summary = " ".join([str(sentence) for sentence in summarizer(parser.document, 3)])
print("TEXT SUMMARY:")
print(summary)
```
## speech to text
```py
import speech_recognition as sr
from textblob import TextBlob
r = sr.Recognizer()
audio_file = sr.AudioFile(r"test-1.wav")
with audio_file as source:
    audio = r.record(source)
text = r.recognize_google(audio)
blob = TextBlob(text)
sentiment = TextBlob(text).sentiment.polarity
sentiment_label = "happy" if sentiment > 0 else "sad" if sentiment < 0 else "neutral"
print(text)
print(sentiment_label)
```
## regex
```py
#Write RE to extract all Email id’s in the given text.
import re
text = "My email is john@example.com."
email_pattern = r'\w+@\w+\.\w+'
emails = re.findall(email_pattern, text)

print(emails)

#Write RE to extract all mobile numbers.
import re
text = "My phone number is 123-456-7890. 9003363162"
phone_pattern = r'\d{3}-?\d{3}-?\d{4}'
phones = re.findall(phone_pattern, text)

print(phones)

#Write RE to extract the names from the below list which match a certain pattern S u _ _ _ Sunil, Shyam, Ankit, Surjeet, Sumit, Subhi, Surbhi, Siddharth, Sujan
import re
names = ['Sunil', 'Shyam', 'Ankit', 'Surjeet', 'Sumit', 'Subhi', 'Surbhi', 'Siddharth', 'Sujan']
name_pattern = r'Su\w{3}'
matching_names = [name for name in names if re.match(name_pattern, name)]

print(matching_names)

#Write RE matches a string that has 'ab' followed by zero or more 'c'.
import re
text = "abcc abc abccc ab abbbc"
pattern = r'abc*'
matches = re.findall(pattern, text)

print(matches)

#Write RE matches 'a' followed by zero or more copies of the sequence 'bc'
import re
text = "a abcbc abcbcbc ab abcb"
pattern = r'ab?c*'
matches = re.findall(pattern, text)

print(matches)

#Write RE matches 'ab' followed by zero or one 'c'
import re
text = "a abcbcbc abcbcbc ab abcb"
pattern = r'a?bc*'
matches = re.findall(pattern, text)

print(matches)


```
#POS tagging
```py 
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
sentences = [
 "I need a flight from Atlanta.",
 "Everything to permit us.",
 "I would like to address the public on this issue."
"We need your shipping address."
]
for sentence in sentences:
 words = nltk.word_tokenize(sentence)
 pos_tags = nltk.pos_tag(words)
 print(pos_tags)
 ```
 #stemming
 ```py
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
ans = PorterStemmer()
text = "Programming Loving Lovely Kind"
token = nltk.word_tokenize(text)
for i in token:
  print("Stemming for {} is {}".format(i, ans.stem(i)))
  ```
#lemmatization
```py
from traitlets.config.application import T
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
ans = WordNetLemmatizer()
text = "Programming Loving Lovely Kind"
token = nltk.word_tokenize(text)
for i in token:
  print("Lemma for {} is {}".format(i,ans.lemmatize(i)))
  ```

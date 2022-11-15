import re
import spacy, nltk
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
nlp = spacy.load('en_core_web_sm')

english_stopwords = stopwords.words("english")
english_words = words.words()

nltk.download('omw-1.4')
nltk.download('words')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
remove_words = ['cid', 'thank', 'hash', 'may', 'mail', 'subject', 'date','mon', 'tue', 'wed', 'intended', 'recipient', 'please', 'notify', 'numeric']

lemmatized_english_words = set([lemmatizer.lemmatize(w.lower()) for w in english_words])

def regex_result(pattern, replacement, text):
    complied_pattern = re.compile(pattern)
    return complied_pattern.sub(replacement, text)

def remove_specific_set(text):
    for s_tup in [
        ('this e-mail and any attachments hereto including', 'in the absence of a fully signed written contract')
    ]:
        s1, s2 = s_tup
        while True:
            try:
                i1 = text.index(s1)
                i2 = text.index(s2)
                text = ''.join([text[:i1], text[i2 + len(s2):]])
                print("continue")
            except:
                break
    return text

def replace_emails(text):
    return regex_result(r'\w+@[\w.]+', ' mail ', text)
def replace_https(text):
    return regex_result(r'((htt)?ps?://)?(www\.)?(\w+\.)?([\w:/]+)\.([a-z/\w]+)', ' ', text)
def replace_only_numbers(text):
    return regex_result(r' \d+(\.)?(\d+)? ', ' ', text)
def retain_chars(text):
    return regex_result(r"[^\w\s]+", ' ', text)
def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in english_stopwords])
def remove_crap_text(text, lemmatized_relevant_english_words):
    tokenized = nltk.wordpunct_tokenize(text)
    #return ' '.join([word for word in tokenized if word in lemmatized_relevant_english_words and word in keep_words])
    #return ' '.join([word for word in tokenized if word not in remove_words])
    return ' '.join([word for word in tokenized if word in lemmatized_relevant_english_words and word not in remove_words])
def retain_non_nums(text):
    return regex_result(r"\d+", ' ', text)
def retain_more_than_two_chars(text):
    return ' '.join([w for w in text.split() if len(w)>3])
def merge_repeated_words(text):
    word_list = text.split()
    if len(word_list)==0: return ""
    prev = word_list[0]
    result = []
    result.append(prev)
    for i in range(1, len(word_list)):
        curr = word_list[i]
        if prev != curr:
            result.append(curr)
        prev = curr
    return ' '.join(result) 
def retain_pos_ents(text):
    doc = nlp(text)
    res = []
    for d in doc:
        #if d.pos_ not in ['INTJ', 'AUX', 'NUM', 'DET'] and d.ent_type_ not in ['PERSON', 'GPE','TIME', 'ORDINAL', 'DATE']:
        if d.pos_ not in ['INTJ', 'AUX', 'NUM', 'DET'] and d.ent_type_ not in ['PERSON', 'GPE','TIME']:
            res.append(d.text)
    return ' '.join(res)

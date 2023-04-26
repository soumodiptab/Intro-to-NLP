
import torchtext
from torchtext.data.utils import get_tokenizer
import re
from cleantext import clean
import spacy

class DataCleaner:
    def __init__(self):
        self.tokenizer = get_tokenizer('basic_english')
        self.nlp = spacy.load('en_core_web_sm', disable = ['ner', 'tagger', 'parser'])
        self.stopwords = self.nlp.Defaults.stop_words

    def replace_dates(self,text):
        date_format_a = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', ' <DATE> ', text)
        date_format_b = re.sub(
            r'[A-Za-z]{2,8}\s\d{1,2},?\s\d {4}', ' <DATE> ', date_format_a)
        date_format_c = re.sub(
            r'\d{2} [A-Z][a-z]{2,8} \d{4}', ' <DATE> ', date_format_b)
        return date_format_c

    def replace_hash_tags(self,text):
        return re.sub(r'(\s|^)#(\w+)', ' <HASHTAG> ', text)

    def remove_special_characters(self,text):
            # remove all special characters 
        return re.sub(r'[^A-Za-z0-9\s]', ' ', text)

    def remove_extra_spaces(self,text):
        return re.sub(r'\s{2,}', ' ', text)

    def replace_hyphenated_words(self,text):
        # replace hyphenated words with words seperated by space
        return re.sub(r'(\w+)-(\w+)', r' \1 \2 ', text)

    def clean_text(self,line):
        # line = line.strip()
        # line = re.sub(r'<|>', ' ', line)
        # line = self.replace_dates(line)
        # line = self.replace_hyphenated_words(line)
        # line = self.replace_hash_tags(line)
        # # remove < and > from the text
        # line = clean(line, no_emoji=True,
        #             no_urls=True,
        #             no_emails=True,
        #             no_phone_numbers=True,
        #             no_currency_symbols=True,           
        #             replace_with_url=" <URL> ",
        #             replace_with_email=" <EMAIL> ",
        #             replace_with_phone_number=" <PHONE> ",
        #             replace_with_currency_symbol=" <CURRENCY> ",
        #             lower=True)
        line = self.remove_special_characters(line)
        line = clean(line,no_numbers=True,
                     no_digits=True,
                     no_punct=True,
                     replace_with_number=" <NUMBER> ",
                     replace_with_digit=" ",
                     replace_with_punct="",
                     lower=True)
        line = self.remove_extra_spaces(line)
        tokens=self.tokenizer(line)
        return " ".join(tokens)
    
    def remove_stopwords(self,text):
        tokens = self.tokenizer(text)
        return " ".join([token for token in tokens if token not in self.stopwords])
    
    def lemmatize(self,text):
        doc = self.nlp(text)
        return " ".join([token.lemma_ for token in doc])
    
    def process(self,text):
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize(text)
        return text
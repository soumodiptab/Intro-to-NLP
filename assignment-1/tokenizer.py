"""
@Author: Soumodipta Bose
@Date: 10-02-2023
"""
import re
import os
from nltk.tokenize import sent_tokenize

class Tokenizer:
    """ Tokenizer class to tokenize the given text
    """

    def __init__(self):
        pass

    def replace_new_lines(self, text):
        return re.sub(r'\n', ' ', text)

    def to_uppercase(self, text):
        return text.upper()

    def to_lowercase(self, text):
        return text.lower()
    # replace from text
    # ---------------------------------------------------------------------------------------------------------------

    def replace_email(self, text):
        return re.sub(r'\S+@\S+', ' <EMAIL> ', text)

    # create a function to replace all dates with <DATE>

    def replace_dates(self, text):
        date_format_a = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', ' <DATE> ', text)
        date_format_b = re.sub(
            r'[A-Za-z]{2,8}\s\d{1,2},?\s\d{4}', ' <DATE> ', date_format_a)
        date_format_c = re.sub(
            r'\d{2} [A-Z][a-z]{2,8} \d{4}', ' <DATE> ', date_format_b)
        return date_format_c

    def replace_time(self, text):
        pass

    # create a function to replace phone numbers with <MOB>
    def replace_phone_numbers(self, text):
        # regex to recoginze phone numbers
        phone_a = re.sub(
            r'(\+\d*-)?\s?(\d{3})\s?(\d{3})\s?(\d{4})', ' <PHONE> ', text)
        phone_b = re.sub(r'(\+\d*-)?\s?(\d{3})\s?(\d{4})', ' <PHONE> ', phone_a)
        return phone_b

    def replace_punctuation(self, text):
        return re.sub(r'[^\w\s]', '', text)

    # replace urls with and without http with <URL>
    def replace_urls(self, text):
        return re.sub(r'(https?:)?(www\.)?(\S+)(\.\S+)', ' <URL> ', text)

    def replace_hash_tags(self, text):
        return re.sub(r'(\s|^)#(\w+)', ' <HASHTAG> ', text)

    def replace_hyphenated_words(self, text):
        # replace hyphenated words with words seperated by space
        return re.sub(r'(\w+)-(\w+)', r'\1 \2', text)

    def replace_concurrent_punctuation(self, text):
        # replace concurrent punctuation with single punctuation
        return re.sub(r'(!|"|\#|\$|%|&|\'|\(|\)|\*|\+|,|-|\.|\/|:|;|<|=|>|\?|@|\[|\\|\]|\^|_|‘|\{|\||\}|~){2,}', r' ', text)

    # remove from text :
    # ---------------------------------------------------------------------------------------------------------------
    # remove spaces with size more than 1
    
    def remove_extra_spaces(self, text):
        return re.sub(r'\s{2,}', ' ', text)

    # Need to improve regex for footnote3
    def remove_footnotes(self, text):
        foot1 = re.sub(r'\[\s?\d+\s?\]', '', text)
        foot2 = re.sub(r'—\s[IVXLCDM]+\s—', '', foot1)
        foot3 = re.sub(r'\d+\.[A-Z]\.(\d+(\.)?)?', '', foot2)
        foot4 = re.sub(r'Section\s\d+\.', '', foot3)
        return foot4

    def add_space_special_characters(self, text):
        # add space before and after special characters
        return re.sub(r'([\*\-\!\`\"\$\&\)\'\(\+\,\-\.\/\:\#\%\;\=\?\@\}\~\[\\\]\^\_\‘\{\|])', r' \1 '," "+ text)

    def add_tag(self, text):
        if text.strip() == "":
            return ""
        return "<BEGIN> " + text.strip() + " <END>\n"

    # Tokenizer function to tokenize the given text
    def tokenize(self, text):
        text = self.replace_new_lines(text)
        text = self.to_lowercase(text)
        text = self.replace_hash_tags(text)
        text = self.replace_urls(text)
        text = self.replace_email(text)
        text = self.replace_dates(text)
        text = self.replace_concurrent_punctuation(text)
        text = self.replace_phone_numbers(text)
        text = self.replace_hyphenated_words(text)
        text = self.remove_footnotes(text)
        text = self.add_space_special_characters(text)
        text = self.remove_extra_spaces(text)
        text = self.add_tag(text)
        #text = self.replace_punctuation(text)
        #text = re.sub(r'[^a-z0-9\s]', '', text)
        return text
    def tokenize2(self,text):
        text = self.replace_new_lines(text)
        text = self.to_lowercase(text)
        text = self.replace_hash_tags(text)
        text = self.replace_urls(text)
        text = self.replace_email(text)
        text = self.replace_dates(text)
        text = self.replace_concurrent_punctuation(text)
        text = self.replace_phone_numbers(text)
        text = self.replace_hyphenated_words(text)
        text = self.remove_footnotes(text)
        text = self.add_space_special_characters(text)
        text = self.remove_extra_spaces(text)
        #text = self.replace_punctuation(text)
        #text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split(" ")

# Util functions to save and read from file


def save_to_file(file_name, text):
    with open(file_name, "w") as f:
        for line in text:
            f.write(line)


def read_from_file(file_name):
    with open(file_name, "r") as f:
        text = f.read()
    return sent_tokenize(text)

def load_from_file(file_name):
    with open(file_name, "r") as f:
        text = f.readlines()
    return text

def build_vocab(file_name, text_lines):
    # build vocabulory and save to file
    vocab = set()
    for text in text_lines:
        vocab.update(set(text.split()))
    vocab = sorted(vocab)
    save_to_file(file_name, vocab)


if __name__ == "__main__":
    current_dir = os.getcwd()
    text_lines = read_from_file("corpora/Pride and Prejudice - Jane Austen.txt")
    tokenizer = Tokenizer()
    processed_text = []
    for text in text_lines:
        proc_txt = tokenizer.tokenize(text.strip())
        if proc_txt.strip() != "":
            processed_text.append(proc_txt)
    save_to_file("clean_corpora/Pride and Prejudice - cleaned.txt", processed_text)

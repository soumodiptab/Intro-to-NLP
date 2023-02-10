"""
@Author: Soumodipta Bose
@Date: 10-02-2023
"""
import re
PATH = "./assignment-1/corpora/Pride and Prejudice - Jane Austen.txt"


class Tokenizer:
    """ Tokenizer class to tokenize the given text
    """

    def __init__(self):
        pass

    def to_uppercase(self, text):
        return text.upper()

    def to_lowercase(self, text):
        return text.lower()
    # replace from text
    # ---------------------------------------------------------------------------------------------------------------

    def replace_email(self, text):
        return re.sub(r'\S+@\S+', '<EMAIL>', text)

    # create a function to replace all dates with <DATE>

    def replace_dates(self, text):
        date_format_a = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '<DATE>', text)
        date_format_b = re.sub(
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s\d{1,2},?\s\d{4}', '<DATE>', date_format_a)
        date_format_c = re.sub(
            r'\d{2} [A-Z][a-z]{2,8} \d{4}', '<DATE>', date_format_b)
        return date_format_c

    def replace_time(self, text):
        pass

    # create a function to replace phone numbers with <MOB>
    def replace_phone_numbers(self, text):
        # regex to recoginze phone numbers
        phone_a = re.sub(
            r'(\+\d*-)?\s?(\d{3})\s?(\d{3})\s?(\d{4})', '<PHONE>', text)
        phone_b = re.sub(r'(\+\d*-)?\s?(\d{3})\s?(\d{4})', '<PHONE>', phone_a)
        return phone_b

    def replace_punctuation(self, text):
        return re.sub(r'[^\w\s]', '', text)
    # replace urls with and without http with <URL>

    def replace_urls(self, text):
        return re.sub(r'(https?:)?(www\.)?(\S+)(\.\S+)', '<URL>', text)

    def replace_hash_tags(self, text):
        return re.sub(r'(\s|^)#(\w+)', '<HASHTAG>', text)

    def replace_hyphenated_words(self, text):
        #replace hyphenated words with words seperated by space
        return re.sub(r'(\w+)-(\w+)', r'\1 \2', text)


    # remove from text :
    # ---------------------------------------------------------------------------------------------------------------
    # remove spaces with size more than 1
    def remove_spaces(self, text):
        return re.sub(r'\s{2,}', ' ', text)

    # Need to improve regex for footnote3
    def remove_footnotes(self, text):
        foot1 = re.sub(r'\[\s?\d+\s?\]', '', text)
        foot2 = re.sub(r'—\s[IVXLCDM]+\s—', '', foot1)
        foot3 = re.sub(r'\d+\.[A-Z]\.(\d+(\.)?)?', '', foot2)
        foot4 = re.sub(r'Section\s\d+\.', '', foot3)
        return foot4

    def add_tag(self, text):
        return "<BEGIN> " + text.strip() + " <END>\n"
    # Tokenizer function to tokenize the given text

    def tokenize(self, text):
        text = self.replace_hash_tags(text)
        text = self.replace_urls(text)
        #text = self.replace_punctuation(text)
        #text = re.sub(r'[^a-z0-9\s]', '', text)
        return text



# Util functions to save and read from file


def save_to_file(file_name, text):
    with open(file_name, "w") as f:
        f.write(text)


def read_from_file(file_name):
    with open(file_name, "r") as f:
        text = f.readlines()
    return text


if __name__ == "__main__":
    text_lines = read_from_file(
        "./assignment-1/corpora/test1.txt")
    tokenizer = Tokenizer()
    for text in text_lines:
        text = tokenizer.tokenize(text)
        print(text)
    pass

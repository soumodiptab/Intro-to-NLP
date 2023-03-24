import json
import pandas as pd
import torchtext
from torchtext.data.utils import get_tokenizer
import re
from tqdm import tqdm
from cleantext import clean
tokenizer = get_tokenizer('basic_english')


def replace_dates(text):
        date_format_a = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', ' <DATE> ', text)
        date_format_b = re.sub(
            r'[A-Za-z]{2,8}\s\d{1,2},?\s\d {4}', ' <DATE> ', date_format_a)
        date_format_c = re.sub(
            r'\d{2} [A-Z][a-z]{2,8} \d{4}', ' <DATE> ', date_format_b)
        return date_format_c

def replace_hash_tags(text):
        return re.sub(r'(\s|^)#(\w+)', ' <HASHTAG> ', text)

def remove_special_characters(text):
        # remove all special characters 
        return re.sub(r'[^A-Za-z0-9\s]', ' ', text)

def remove_extra_spaces(text):
        return re.sub(r'\s{2,}', ' ', text)

def replace_hyphenated_words(text):
        # replace hyphenated words with words seperated by space
        return re.sub(r'(\w+)-(\w+)', r' \1 \2 ', text)

def create_sampled_dataset(input_file,line_count,atleast=10):
    lines = []
    ctr= 0
    print('Cleaning :',input_file)
    with open(input_file, 'r') as f:
        for line in tqdm(f.readlines()):
            line = json.loads(line.strip())['reviewText']
            line = line.strip()
            line = re.sub(r'<|>', ' ', line)
            line = replace_dates(line)
            line = replace_hyphenated_words(line)
            line = replace_hash_tags(line)
            # remove < and > from the text
            line = clean(line, no_emoji=True,
                        no_urls=True,
                        no_emails=True,
                        no_phone_numbers=True,
                        no_currency_symbols=True,           
                        replace_with_url=" <URL> ",
                        replace_with_email=" <EMAIL> ",
                        replace_with_phone_number=" <PHONE> ",
                        replace_with_currency_symbol=" <CURRENCY> ",
                        lower=True)
            line = remove_special_characters(line)
            line = clean(line,no_numbers=True,no_digits=True,no_punct=True, replace_with_number=" <NUMBER> ",replace_with_digit=" ",replace_with_punct="")
            line = remove_extra_spaces(line)
            tokens=tokenizer(line)
            if len(tokens)>atleast:
                lines.append(tokens)
                ctr+=1
            if ctr >= line_count:
                break
    return lines


def save_data(filename, lines):
    # Save the data to a file
    with open(filename, 'w')as f:
        print('Saving the data to :',filename)
        for line in tqdm(lines):
            line = ' '.join(line)
            f.write(line.strip()+'\n')


def get_vocab(file_path):
    vocab = set()
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            tokens = line.split()
            vocab.update(tokens)
    vocab=sorted(vocab)
    return vocab

def clean_corpus(corpus_file_path,save_file_path,sample,sentences,min_word_count):
    try:
        print('Cleaning the data')
        print('Corpus file path : {} Sentences : {} Min. no. of words : {}'.format(corpus_file_path,sentences,min_word_count))
        lines = create_sampled_dataset(corpus_file_path,sentences,min_word_count)
        save_data(save_file_path, lines)
        print('Data Saved to :',save_file_path)
        vocab = get_vocab(save_file_path)
        print('Vocab Size :',len(vocab))
    except Exception as e:
        print('Error in cleaning the data')
        print(e)
        exit(1)
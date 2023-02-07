import re

class Tokenizer:
    def __init__(self):
        pass
    def cleaning(self,text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text

if __name__ == `__main__`:
    pass
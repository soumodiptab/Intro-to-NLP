import tokenizer as tk
import argparse
import os
import pickle
import numpy as np
import time
ROLL_NO = "2021201086"


def decorate():
    print("=================================================================================================")

def error(msg):
    decorate()
    print("[ERROR] : " + msg)



def info(msg):
    decorate()
    print("[INFO] : " + msg)


class Ngram:
    def __init__(self):
        pass

    def get_ngram(self, text_line, n):
        text_line = text_line.strip()
        ngram = []
        tokens = text_line.split(' ')
        if len(tokens) == 0:
            return None
        tokens = ["<BEGIN>" for _ in range(n-2)] + tokens
        for i in range(len(tokens)-n+1):
            ngram.append(" ".join(tokens[i:i+n]))
        return ngram

    def __generate_freq_table(self, text_lines, n):
        freq_table = {}
        for text in text_lines:
            ngram = self.get_ngram(text, n)
            if ngram is not None:
                for token in ngram:
                    if token in freq_table:
                        freq_table[token] += 1
                    else:
                        freq_table[token] = 1
        return freq_table

    def construct_freq_table(self, text_lines, n, threshold=10):
        freq_table = {}
        for k in range(1, n+1):
            freq_table[k] = self.__generate_freq_table(text_lines, k)
        freq_table[1]["<#>"] = 0
        key_set = list(freq_table[1].keys())
        for key in key_set:
            if freq_table[1][key] < threshold:
                freq_table[1]["<#>"] += freq_table[1][key]
                freq_table[1].pop(key)
        return freq_table


class Smoothing:
    def __init__(self):
        pass

    def get_perplexity(self, model, test_data):
        pass


class NgramModel:
    def __init__(self, n):
        self.n = n
        self.ngram = Ngram()
        self.cache = {}

    def is_ngram_present(self,n,ngram):
        if ngram in self.freq_table[n]:
            return True
        return False

    def count_size(self,n):
        return len(self.freq_table[n])

    def get_ngram_freq(self,n, ngram):
        return self.freq_table[n][ngram]

    def count_ngram_freq(self, ngram):
        # freq count( ngram)
        tokens = ngram.split(" ")
        if len(tokens) > 0:
            if ngram in self.freq_table[len(tokens)]:
                return self.freq_table[len(tokens)][ngram]
        return 0
    def count_ngram_history_freq(self,history):
        # freq count( history + variable word)
        # cache_key = "history_freq_"+history
        # if cache_key in self.cache:
        #     return self.cache[cache_key]
        
        gram = len(history.split(" "))+1
        total =0
        for key in self.freq_table[gram]:
            if key.startswith(history):
                total += self.freq_table[gram][key]
        # self.cache[cache_key] = total
        return total
    def count_ngram_history(self,history):
        # count( history + variable word)
        # cache_key = "history_"+history
        # if cache_key in self.cache:
        #     return self.cache[cache_key]
        gram = len(history.split(" "))+1
        total =0
        for key in self.freq_table[gram]:
            if key.startswith(history):
                total += 1
        # self.cache[cache_key] = total
        return total

    def count_ngram_current(self,gram,current):
        # count( variable history + current)
        # cache_key = "current_"+gram+"_"+current
        # if cache_key in self.cache:
        #     return self.cache[cache_key]
        total = 0
        for key in self.freq_table[gram]:
            if key.endswith(current):
                total += 1
        # self.cache[cache_key] = total
        return total

    def train(self, train_data):
        self.freq_table = self.ngram.construct_freq_table(train_data, self.n)

    def get_perplexity(self, test_sentence, smoothing: Smoothing):
        perplexity_scores = []
        ngram_tokens = self.ngram.get_ngram(test_sentence, self.n)
        for token in ngram_tokens:
            score = smoothing.get_perplexity(self, token)
            perplexity_scores.append(score)
        N = len(ngram_tokens)
        perplexity_score = pow(1/np.prod(perplexity_scores), 1/N)
        return perplexity_score



class WittenBell(Smoothing):
    def __init__(self):
        super().__init__()

    def __P_wb(self,model: NgramModel,n, history,current):
        # Probability of current word given history
        if n == 1:
            if model.is_ngram_present(n,current):
                return model.get_ngram_freq(n,current)/model.count_size(n)
            return 1/len(model.freq_table[1])
        try:
            LAMBDA = model.count_ngram_history(history)/(
                model.count_ngram_history(history) +model.count_ngram_history_freq(history)
            )
        except:
            new_history = " ".join(history.split(" ")[1:])
            return self.__P_wb(model,n-1,new_history,current)
        P_mle = model.count_ngram_freq(history+" "+current)/model.count_ngram_history_freq(history)
        new_history = " ".join(history.split(" ")[1:])
        P_backoff = self.__P_wb(model,n-1,new_history,current)
        return LAMBDA*P_mle + (1-LAMBDA)*P_backoff

    def get_perplexity(self, model: NgramModel, ngram_token):
        tokens = ngram_token.split(" ")
        history= " ".join(tokens[:-1])
        current = tokens[-1]
        return self.__P_wb(model,len(tokens),history,current)


class KneserNey(Smoothing):
    def __init__(self):
        super().__init__()

    def __P_kn(self,model: NgramModel,n, history,current,higher_order=False,d=0.75):
        # Probability of current word given history
        if not model.is_ngram_present(1,current):
            return d/model.get_ngram_freq(1,"<#>")
        if n == 1:
            return (d / model.get_ngram_freq(1,"<#>")) + ((1-d)/model.count_size(1))
        try:
            if higher_order:
                ngram = " ".join([history,current])
                FIRST_TERM =max(model.count_ngram_freq(ngram)-d,0)/model.count_ngram_history_freq(history)
            else:
                FIRST_TERM =max(model.count_ngram_current(n,current) - d,0)/model.count_size(n)
        except:
            FIRST_TERM = 0
        try:
            LAMBDA = (d/model.count_ngram_history_freq(history))*model.count_ngram_history(history)
        except:
            LAMBDA = (d/model.get_ngram_freq(1,"<#>"))
            return LAMBDA
        new_history = " ".join(history.split(" ")[1:])
        SECOND_TERM = self.__P_kn(model,n-1,new_history,current)
        return FIRST_TERM + LAMBDA * SECOND_TERM


    def get_perplexity(self, model: NgramModel, ngram_token):
        tokens = ngram_token.split(" ")
        history= " ".join(tokens[:-1])
        current = tokens[-1]
        return self.__P_kn(model,len(tokens),history,current,True)


def test_train_split(data, test_distribution=0.2, flag=False):
    test_size = 1000
    if flag:
        test_size = n*test_distribution
    #seq =int(time.time())
    seq = 12
    np.random.seed(seq)
    n = len(data)
    idx_list = np.random.choice(n, int(test_size), replace=False)
    train_data = []
    test_data = []
    for i in range(len(data)):
        if i in idx_list:
            test_data.append(data[i])
        else:
            train_data.append(data[i])
    return train_data, test_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", help="Ngram value")
    parser.add_argument("smoothing", help="Smoothing technique to be used")
    parser.add_argument("corpus", help="Path to the corpus file")
    args = parser.parse_args()
    smoothing_technique = args.smoothing
    CORPUS_PATH = args.corpus
    N = int(args.n)
    corpus_name = CORPUS_PATH.split("/")[-1].split(".")[0]
    if not os.path.exists(CORPUS_PATH):
        error('Corpus file does not exist.')
        exit(1)
    CLEAN_CORPORA_PATH = os.path.join("clean_corpora", corpus_name+".txt")
    clean_text_lines = []
    if not os.path.exists(CLEAN_CORPORA_PATH):
        error('Clean corpus file does not exist.')
        info('Creating clean corpus file.')
        tokenizer = tk.Tokenizer()
        text_lines = tk.read_from_file(CORPUS_PATH)
        clean_text_lines = []
        for text in text_lines:
            clean_text_line = tokenizer.tokenize(text)
            if clean_text_line.strip() != "":
                clean_text_lines.append(clean_text_line)
        tk.save_to_file(CLEAN_CORPORA_PATH, clean_text_lines)
        info('Clean corpus file created.')
    else:
        clean_text_lines = tk.read_from_file(CLEAN_CORPORA_PATH)
    info('loading clean corpus file.')
    train, test = test_train_split(clean_text_lines)
    MODEL_PATH = os.path.join(".", "models", ROLL_NO+"_"+corpus_name+".pkl")
    if not os.path.exists(MODEL_PATH):
        info('Creating Ngram Model using training data.')
        model = NgramModel(N)
        model.train(train)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        info('Ngram model created.')
    else:
        info('Loading Ngram model.')
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    info('Ngram model loaded.')
    LM = input("Enter the name for score TXT file (for which LM): ")
    if smoothing_technique == "k":
        k = KneserNey()
        info('Calculating perplexity using KneserNey smoothing for training data.')
        perplexity_scores_train = []
        PERPLEXITY_SCORE_TRAIN_PATH = os.path.join(".", "scores", ROLL_NO+"_LM"+LM+"_train-perplexity.txt")
        with open(PERPLEXITY_SCORE_TRAIN_PATH, 'w') as f:
            for text in train:
                perplexity_score = model.get_perplexity(text, k)
                perplexity_scores_train.append(perplexity_score)
                # info(text.strip() + " :: " + str(perplexity_score))
                f.write(text.strip() + " :: " + str(perplexity_score) +"\n")
            f.write("Average Perplexity for training data: " + str(np.mean(perplexity_scores_train)))
        info('Perplexity calculated for training data. Saving to file.')
        info ('Average Perplexity for training data: ' + str(np.mean(perplexity_scores_train)))
        info('Calculating perplexity using KneserNey smoothing for training data.')
        PERPLEXITY_SCORE_TEST_PATH = os.path.join(".", "scores", ROLL_NO+"_LM"+LM+"_test-perplexity.txt")
        perplexity_scores_test =[]
        with open(PERPLEXITY_SCORE_TEST_PATH, 'w') as f:
            for text in test:
                perplexity_score = model.get_perplexity(text, k)
                perplexity_scores_test.append(perplexity_score)
                # info(text.strip() + " :: " + str(perplexity_score))
                f.write(text.strip() + ":: " + str(perplexity_score) +"\n")
            f.write("Average Perplexity for test data: " + str(np.mean(perplexity_scores_test)))
        info('Perplexity calculated. Saved to file.')
        info ('Average Perplexity for test data: ' + str(np.mean(perplexity_scores_test)))

    elif smoothing_technique == "w":
        w = WittenBell()
        # info('Calculating perplexity using Witten Bell smoothing for training data.')
        # perplexity_scores_train = []
        # PERPLEXITY_SCORE_TRAIN_PATH = os.path.join(".", "scores", ROLL_NO+"_LM"+LM+"_train-perplexity.txt")
        # with open(PERPLEXITY_SCORE_TRAIN_PATH, 'w') as f:
        #     for text in train:
        #         perplexity_score = model.get_perplexity(text, w)
        #         perplexity_scores_train.append(perplexity_score)
        #         # info(text.strip() + " :: " + str(perplexity_score))
        #         f.write(text.strip() + " :: " + str(perplexity_score) +"\n")
        #     f.write("Average Perplexity for training data: " + str(np.mean(perplexity_scores_train)))
        # info('Perplexity calculated for training data. Saving to file.')
        # info ('Average Perplexity for training data: ' + str(np.mean(perplexity_scores_train)))
        info('Calculating perplexity using Witten Bell smoothing for training data.')
        PERPLEXITY_SCORE_TEST_PATH = os.path.join(".", "scores", ROLL_NO+"_LM"+LM+"_test-perplexity.txt")
        perplexity_scores_test =[]
        i=1
        with open(PERPLEXITY_SCORE_TEST_PATH, 'w') as f:
            for text in test:
                perplexity_score = model.get_perplexity(text, w)
                perplexity_scores_test.append(perplexity_score)
                # print(str(i) + " :: " + str(perplexity_score))
                f.write(text.strip() + ":: " + str(perplexity_score) +"\n")
                i+=1
            f.write("Average Perplexity for test data: " + str(np.mean(perplexity_scores_test)))
        info('Perplexity calculated. Saved to file.')
        info ('Average Perplexity for test data: ' + str(np.mean(perplexity_scores_test)))
        
    else:
        error("Incorrect smoothing technique.")
        exit(1)

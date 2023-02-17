
class n_gram_model:
    def get_string(self, arr, i ,length):
        return " ".join(arr[i:i+length])

    def _init_(self, train_line_vector_preprocessed, n_gram, THRESHOLD):
        self.table = {}
        self.THRESHOLD = THRESHOLD
        self.UKN = "<UKN>"
        self.train_size = 0
        self.start_with_dict = {}
        self.ends_with_dict = {}

        for line in train_line_vector_preprocessed:
            word_vector = line.split()
            for i in range(0,len(word_vector)-n_gram+1):
                string = self.get_string(word_vector,i,n_gram)
                if string not in self.table:
                    self.table[string] = 0
                self.table[string] += 1 
            self.train_size += len(word_vector)
        
        for key, value in self.table.items():
            word_vector = key.split()
            curr = ""
            for word in word_vector:
                curr += " " + word 
                if curr[1:] not in self.start_with_dict:
                    self.start_with_dict[curr[1:]] = 0
                self.start_with_dict[curr[1:]] += 1 

        for key, value in self.table.items():
            word_vector = key.split()
            word_vector.reverse()
            curr = ""
            for word in word_vector:
                curr += " " + word 
                if curr[1:] not in self.ends_with_dict:
                    self.ends_with_dict[curr[1:]] = 0
                self.ends_with_dict[curr[1:]] += 1 
        if n_gram == 1:
            remove_list = ()
            for k,v in self.table.items():
                if v <= THRESHOLD:
                    remove_list += (k,)
   
            for k in remove_list:
                if self.UKN not in self.table:
                        self.table[self.UKN] = 0
                self.table[self.UKN] += self.table[k]
                del self.table[k]
                
                    
    def get_count(self, word_arr):
        string = ' '.join(word_arr)
        if string in self.table:
            return self.table[string]
        else:
            return 0
    
    def get_type_count(self, word_arr):
        string = ' '.join(word_arr)
        if not string:
            return len(self.table)
class kenserNey:
    def __init__(self, n_gram_model, n):
        self.d = 0.75
        self.n = n
        self.THRESHOLD  = 1
        self.UKN = "<UKN>"
        self.n_gram_model = n_gram_model
        self.n_gram_table = {}
        self.vocab_size = 0

    def train(self, train_line_vector_preprocessed):    
        for i in range(1, self.n+1):
            self.n_gram_table[i] = n_gram_model(train_line_vector_preprocessed, i, self.THRESHOLD)
        self.vocab_size = self.n_gram_table[1].get_size()

    def get_continuation_count(self, prev_words_arr, curr_word_arr, i):
        if i == self.n:
            numerator = self.n_gram_table[i].get_count(prev_words_arr + curr_word_arr)
            denominator = self.n_gram_table[i-1].get_count(prev_words_arr)
        else:
            numerator = self.n_gram_table[i].get_preciding_count(curr_word_arr) 
            denominator = self.n_gram_table[i].get_size()
        return numerator, denominator 

    def get_first_term(self, prev_words_arr, curr_word, i):
        nn, dd = self.get_continuation_count(prev_words_arr, curr_word, i)
        try: 
            return max(nn - self.d,0) / dd
        except ZeroDivisionError:
            return 0


    def get_lambda(self, prev_words_arr, i):
        #Note that lambda is a function of only the string (prev words), not of the final words(n-gram)
        #what if i == 1??
        if i != 1:
            try:
                #print(prev_words_arr)
                return self.d / self.n_gram_table[i-1].get_count(prev_words_arr) * self.n_gram_table[i].get_type_count(prev_words_arr)
            except ZeroDivisionError:
                return self.d
        else:
            return self.d / self.vocab_size

    def get_prob(self, prev_words_arr, curr_word_arr, i):
        if self.n_gram_table[1].get_count(curr_word_arr) == 0:
            try:
                return self.d / self.vocab_size
                #return self.d / self.n_gram_table[1].get_count((self.UKN,))
            except ZeroDivisionError:
                return 0

        if i == 1:
            try:
                return self.get_first_term(prev_words_arr, curr_word_arr , i) + self.d / self.vocab_size
                #return (1 - self.d) / self.n_gram_table[1].get_size() + self.d / self.n_gram_table[1].get_count((self.UKN,))
            except ZeroDivisionError:
                return 0

        firstTerm = self.get_first_term(prev_words_arr, curr_word_arr , i)
        secondTerm = self.get_lambda(prev_words_arr, i)\
                          * self.get_prob(prev_words_arr[1:], curr_word_arr, i-1)
        
        return firstTerm + secondTerm
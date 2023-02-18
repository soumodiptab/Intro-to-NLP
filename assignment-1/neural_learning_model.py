import random
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.models import Sequential,load_model
import tensorflow.keras.utils as ku 
import numpy as np
import math
import re
from nltk import sent_tokenize

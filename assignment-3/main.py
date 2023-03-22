from utils.data_pipeline import DataPipeline
from models.cbow import CBOWNEG

dataset = DataPipeline('data/processed_data/corpus_cleaned.txt')
print('Dataset Loaded')
model = CBOWNEG(len(dataset.vocab),200)
print('Model Training :')
model.trainer(dataset,batch_size=128,epochs=10,lr=0.001,print_every=1,checkpoint_every=2)
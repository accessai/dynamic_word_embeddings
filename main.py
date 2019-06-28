import torch
import torch.utils.data

from dataset import SkipGramDataSet
from train import Trainer
from skip_gram import SkipGramModel

skip_gram_dataset = SkipGramDataSet('./data/train_data.npy', vocab_path='./data/id2word.pkl')
data_loader = torch.utils.data.DataLoader(dataset=skip_gram_dataset,
                                               batch_size=5000,
                                               shuffle=True)

model = SkipGramModel(skip_gram_dataset.vocab_length,m=40)
trainer = Trainer(model, data_loader)
model = trainer.train(epochs=10, lr=0.001)

torch.save(model, 'model.ckpt')
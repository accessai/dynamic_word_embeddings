import os
import argparse

import torch.utils.data

from dataset import SkipGramDataSet
from train import Trainer
from skip_gram import SkipGramModel

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', 'Data directory', default='./data')
parser.add_argument('--embed_dim', 'Embedding dimension', type=int, default=100)
parser.add_argument('--batch_size', 'Batch Size', type=int, default=10)
parser.add_argument('--lr', 'Learning rate', type=float, default=0.001)
parser.add_argument('--epochs', 'Epochs', type=int, default=5)
parser.add_argument('--device', 'cpu/gpu', default='cpu')

args = parser.parse_args()
vocab_path = os.path.join(args.get('data_dir'), 'i2word.pkl')
train_data_path = os.path.join(args.get('data_dir'),'train_data.npy')
embed_dim = args.get('embed_dim')
batch_size = args.get('batch_size')
epochs = args.get('epochs')
learning_rate = args.get('lr')
device = 'cpu' if args.get('device') == 'cpu' else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

skip_gram_dataset = SkipGramDataSet(train_data_path, vocab_path=vocab_path)
data_loader = torch.utils.data.DataLoader(dataset=skip_gram_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

model = SkipGramModel(skip_gram_dataset.vocab_length,m=embed_dim)
trainer = Trainer(model, data_loader, device)
model = trainer.train(epochs=epochs, lr=learning_rate)

torch.save(model, 'model.ckpt')
# Dynamic Word Embeddings
Study of semantic evolution of words over time


#### Download data and preprocess:
```
python -m download_data
python -m preprocess_data
```

#### Training
```
python -m main --data_dir=./data \
               --embed_dim=100 \
               --batch_size=1000 \
               --lr=0.001 \
               --epochs=5 \
               --device=cpu
```

# References:
https://arxiv.org/pdf/1702.08359.pdf


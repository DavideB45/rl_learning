from dynamics.blocks_tr import Transformer, PositionalEncoding
import torch as t

EMB = 2
LEN = 3
BAT = 2
enc = PositionalEncoding(emb_size=EMB, dropout=0, max_len=14)

print('Seq Bat Emb')
standard = enc(t.zeros([LEN, BAT, EMB]))
for i in range(LEN):
	print(standard[i, :])

enc = PositionalEncoding(emb_size=EMB, dropout=0, max_len=14, batch_first=True)
print('Bat Seq Emb')
bf = enc(t.zeros([BAT, LEN, EMB]))
for i in range(LEN):
	print(bf[:, i, :])

from dynamics.transformer import TransformerArc
from helpers.general import best_device

model = TransformerArc(
	in_size=3,
	emb_size=10,
	max_seq_len=20,
	n_heads=2,
	n_transformer=3,
	dropout=0.1,
	device=best_device()
)

print(model)
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
counter = Counter()

lines = [line for line in train_iter]

for line in train_iter:
    counter.update(tokenizer(line))
vocab = Vocab(counter)

print(lines[10])
print(len(lines))

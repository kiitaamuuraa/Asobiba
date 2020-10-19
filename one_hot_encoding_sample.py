import janome.tokenizer
import sklearn.preprocessing 

# 夏目漱石「こころ」の冒頭
# https://www.aozora.gr.jp/cards/000148/files/773_14560.html

d_01 = "私わたくしはその人を常に先生と呼んでいた。"
d_02 = "だからここでもただ先生と書くだけで本名は打ち明けない。"
d_03 = "これは世間を憚はばかる遠慮というよりも、その方が私にとって自然だからである。"
d_04 = "私はその人の記憶を呼び起すごとに、すぐ「先生」といいたくなる。"

tokenizer = janome.tokenizer.Tokenizer()
ds = [d_01, d_02, d_03, d_04]

# ボキャブラリー作成
l = list()
for d in ds:
    l += [token.base_form for token in tokenizer.tokenize(d)]
vocab = list(set(l))

# ワンホットエンコーディング
onehot_vocab = sklearn.preprocessing.label_binarize(vocab,classes=vocab)

# 辞書
onehot_dict = dict(zip(vocab, onehot_vocab))
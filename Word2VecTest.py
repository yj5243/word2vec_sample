# -*- coding: utf-8 -*-

import re
import pandas as pd
import numpy as np
from gensim.models import Word2Vec

# ------------------------------
# -- 문장별 Word2Vec 처리
# ------------------------------
model = Word2Vec.load("model/doc2vec.model")
print("<model", "_" * 100)
print(model)
print("<model", "_" * 100)

model.save("doc2vec.model")
print("-"*100)
for word, score in model.most_similar(positive=[  "하나"] ):
    print(word)

print("-"*100)
for word, score in model.most_similar(positive=[  "1"] ):
    print(word)


# sentence = "파이썬에서 형태소파이썬에서 형태소 파이썬에서 형태소 분석하기"
# tokenized_contents = [];
#
# mecab = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ko-dic')
# mecabParse = mecab.parse("파이썬에서 형태소 분석하기 ").split("\n")
#
#
# for word in mecabParse:        # Second Example
#     wordAnalys = word.split("\t")
#     if wordAnalys[0] != "EOS" and len(wordAnalys[0]) > 0:
#         tokenized_contents.append(wordAnalys[0])
#
#
# print(tokenized_contents);

# -*- coding: utf-8 -*-

import re
import pandas as pd
import numpy as np
from gensim.models import Word2Vec

# ------------------------------
# -- 파일 읽기
# ------------------------------
# file = open("data/all_wiki_jira_landNewstitle.txt", "r", encoding='utf8')
file = open("data/moby.txt", "r", encoding='utf8')
moby_dick = file.read()
print(moby_dick)

print("<raw_doc", "_" * 100)

# ------------------------------
# -- 문장별로 Split 처리
# ------------------------------
moby_dick = re.split("[\n\.?]", moby_dick)
print(moby_dick)

print("<split_doc", "_" * 100)

# ------------------------------
# -- 공백/빈 리스트 제거
# ------------------------------
while ' ' in moby_dick:
    moby_dick.remove(' ')
    moby_dick.remove('')

    print(moby_dick)

print("<remove_blank_doc", "_" * 100)

# ------------------------------
# -- 데이터프레임에 저장
# ------------------------------
df_Mobydic = pd.DataFrame()
df_Mobydic['sentences'] = np.asarray(moby_dick)

print(df_Mobydic)

print("<df_doc", "_" * 100)

# ------------------------------
# -- 데이터프레임 문장별 Split
# ------------------------------
df_Mobydic["separates"] = df_Mobydic["sentences"].apply(lambda x: x.replace(",", ""))
df_Mobydic["separates"] = df_Mobydic["separates"].apply(lambda x: x.replace(";", ""))
df_Mobydic["separates"] = df_Mobydic["separates"].apply(lambda x: x.replace("\"", ""))
df_Mobydic["separates"] = df_Mobydic["separates"].apply(lambda x: x.split())

print(df_Mobydic)

print("<df_sep_doc", "_" * 100)

# ------------------------------
# -- 문장별 Word2Vec 처리
# ------------------------------
model = Word2Vec(df_Mobydic["separates"], alpha=0.025, iter=100, workers=8, batch_words=1000, sg=1, size=300, min_count=5)
print("<model", "_" * 100)
print(model)
print("<model", "_" * 100)

model.save("model/moby_doc2vec.model")
print("-"*100)
for word, score in model.most_similar("whale"):
    print(word)

print("-"*100)
for word, score in model.most_similar("fish"):
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

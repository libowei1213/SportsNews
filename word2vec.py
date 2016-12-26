# coding=utf-8

from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import time
import pickle

import json
import jieba

jieba.initialize()


# word2Vec sentences类
class MySentences(object):
    def __init__(self, fileName):
        self.fileName = fileName

    def __iter__(self):
        for line in open(self.fileName, encoding="utf-8"):
            jsonObj = json.loads(line)
            words = jsonObj['contentSeg']
            yield words.split()


# 训练word2Vec模型
def trainModel():
    sentences = MySentences("all_seg.txt")
    model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
    model.save_word2vec_format("word2vec.model", binary=True)


# doc2vec 自定义类
class LabeledLineSentence(object):
    def __init__(self, fileName):
        self.fileName = fileName

    def __iter__(self):
        for line in open(self.fileName, encoding="utf-8"):
            jsonObj = json.loads(line)
            words = jsonObj['contentSeg']
            id = jsonObj['id']
            yield TaggedDocument(words=words.split(), tags=[id])


def trainDocModel():
    documents = LabeledLineSentence("all_seg.txt")
    model = Doc2Vec(documents, size=100, window=8, min_count=5, workers=4)
    model.save("doc2vec.model")


def getMostSimDoc(doc2vecModel, docId, topK=5):
    if docId not in doc2vecModel.docvecs:
        print("不在训练集中")
        return

    sims = doc2vecModel.docvecs.most_similar(docId, topn=topK)
    return list(map(lambda x: x[0], sims))


def getAllDocSims():
    model = Doc2Vec.load("doc2vec.model")
    docList = []
    with open("all_seg.txt", encoding="utf=8") as file:
        for line in file:
            jsonObj = json.loads(line)
            id = jsonObj['id']
            docList.append(id)
    dic = {}
    for doc in docList:
        dic[doc] = getMostSimDoc(model, doc)
    pickle.dump(dic, open("doc_similar_dict.bin", "wb"), True)


if __name__ == '__main__':
    getAllDocSims()

    # 读取模型

    # print(time.time())
    #
    # model = Word2Vec.load_word2vec_format("word2vec.model", binary=True)
    # a = time.time()
    # print(model.most_similar(positive=['哈登'], topn=10))
    # print(time.time() - a)
    # model = Doc2Vec.load("doc2vec.model")
    #
    # print(getMostSimDoc(model, "sohu_475153683"))

    # getAllDocSims()

    # dict = pickle.load(open("doc_similar_dict.bin", "rb"))
    #
    # docList = []
    #
    # for doc in dict.keys():
    #     docList.extend(dict[doc])
    #
    # print(len(set(docList)))

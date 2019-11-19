import os
import re
from nltk.tokenize import word_tokenize
import collections
import pandas as pd
import pickle
import numpy as np
import random
from collections import Counter
import nltk


# word_tokenize = nltk.data.load('/data1/yangh/TextStego/nltk_data-gh-pages/packages/tokenizers/punkt/PY3/english.pickle')

def clean_str(text):
    if re.match('[\u4e00-\u9fa5]+', text) is not None:
        if len(text) % 4 == 3:
            text = text + '。'
        char_list = list(text)
        for i, index in enumerate([x.start() for x in re.finditer('[，。？！：]', text)]):
            if i % 2 == 0:
                char_list[index] = '，'
            else:
                char_list[index] = '。'
        text = ' '.join(char_list)
    else:
        text = re.sub(r"[^A-Za-z0-9(),!?\'\`\"]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip().lower()
    return text


def build_word_dict(dict_dir, vocabulary_size=None):
    dict_path = os.path.join(dict_dir, "word_dict.pickle")
    if os.path.exists(dict_path):
        with open(dict_path, "rb") as f:
            word_dict = pickle.load(f)
        # if vocabulary_size is None or len(word_dict) == vocabulary_size:
        print("use word dictionary at %s, vocabulary size: %d" % (dict_path, len(word_dict)))
        return word_dict
    src_texts = [os.path.join(dict_dir, x) for x in os.listdir(dict_dir) if x.endswith('.txt')]
    contents = []
    for src in src_texts:
        with open(src, 'r', encoding='utf8', errors='ignore') as f:
            contents.extend(f.readlines())
    words = list()
    for content in contents:
        for word in word_tokenize(clean_str(content)):
            words.append(word)

    word_counter = collections.Counter(words).most_common()
    word_dict = dict()
    word_dict["<pad>"] = 0
    word_dict["<unk>"] = 1
    word_dict["<s>"] = 2
    word_dict["</s>"] = 3
    for word, count in word_counter:
        if vocabulary_size is None or len(word_dict) != vocabulary_size:
            word_dict[word] = len(word_dict)

    with open(dict_path, "wb") as f:
        pickle.dump(word_dict, f)
    print("build dictionary from %s, vocabulary size: %d" % (dict_path, len(word_dict)))
    # build_embedding(word_dict, dict_dir)
    return word_dict


def build_word_dataset(train_path, test_path, step, word_dict, document_max_len, label_map=None, up_sample=False):
    if step == "train":
        df = pd.read_csv(train_path, names=["class", "title", "content"])
    else:
        df = pd.read_csv(test_path, names=["class", "title", "content"])

    # Shuffle dataframe
    df = df.sample(frac=1)
    x = list(map(lambda d: word_tokenize(clean_str(d)), df["content"]))
    x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
    x = list(map(lambda d: d[:document_max_len], x))
    x = list(map(lambda d: d + (document_max_len - len(d)) * [word_dict["<pad>"]], x))

    # y = list(map(lambda d: d - 1, list(df["class"])))
    y = list(map(lambda d: d, list(df["class"])))
    if label_map is not None:
        y = [label_map[i] for i in y]
    if step == 'train':
        label_count = list(Counter(y).items())
        label_count.sort(key=lambda d: d[1])
        max_train_sample = round(label_count[-1][1] * 0.8)
        x_train = []
        y_train = []
        x_valid = []
        y_valid = []
        for label, count in label_count:
            index = np.where(np.array(y) == label)[0]
            train_sample = round(len(index) * 0.8)
            index_train = index[:train_sample]
            index_valid = index[train_sample:]
            if up_sample:
                index_train = list(index_train) * (max_train_sample // len(index_train) + 1)
                index_train = random.sample(index_train, max_train_sample)
            x_valid.extend(np.array(x)[index_valid].tolist())
            y_valid.extend(np.array(y)[index_valid].tolist())
            x_train.extend(np.array(x)[index_train].tolist())
            y_train.extend(np.array(y)[index_train].tolist())
        z = list(zip(x_train, y_train))
        random.shuffle(z)
        x_train, y_train = zip(*z)
        # epison = 0.2
        # y_train = [(1-epison)*x + epison/2  for x in y_train]
        return x_train, y_train, x_valid, y_valid
    return x, y


def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]


def save_variable(file_name, variable):
    file_object = open(file_name, "wb")
    pickle.dump(variable, file_object)
    file_object.close()


def build_embedding(word_dict, save_path):
    dict_path = os.path.join(save_path, "embeding.pkl")
    if os.path.exists(dict_path):
        with open(dict_path, "rb") as f:
            word_dict = pickle.load(f)
        return word_dict
    embdict=dict()
    plo=0
    line=0
    with open('/data1/GoogleNews-vectors-negative300.bin','rb')as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode("utf-8", 'ignore')
                if ch ==' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if len(word) != 0:
                tp= np.fromstring(f.read(binary_len), dtype='float32')
                if word in word_dict:
                    embdict[word]=tp.tolist()
                    # if plo%100==0:
                    #     print(plo,line,word)
                    # plo+=1
                    #print(word,tp)
            else:
                f.read(binary_len)
    # In[ ]:
    # print(len(word_dict))
    lister=[0]*len(word_dict)
    for i in embdict.keys():
        tmp = np.array(embdict[i],dtype='float32')
        # if tmp.shape[0] < 200:
        #     tmp = np.array(embdict[0],dtype='float32')
        lister[word_dict[i]]=tmp
    for i in range(len(lister)):
        if isinstance(lister[i], int):
            lister[i] = np.random.uniform(-0.1,0.1,(300,))
    # print("test_lister:",len(lister))
    # import pdb
    # pdb.set_trace()
    # print("data_path:", save_path)
    lister=np.array(lister,dtype='float32')
    save_variable(os.path.join(save_path,'embeding.pkl'), lister)
    return lister

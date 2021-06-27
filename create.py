# %%
from model import doc2vec_model
from model import tfidf_model
from model import bow_model
from preprocess import preprocess_documents
from preprocess import preprocess_document
from multiprocessing import Pool
from time import time
from copy import copy
from random import random
from random import randrange
from os.path import join
from os import listdir

# %%
def create_file_dict(newsgroups_path):
    return {folder: [join(newsgroups_path, folder, file)
                     for file in listdir(join(newsgroups_path, folder))]
            for folder in listdir(newsgroups_path)}


# %%
def create_positive_bags(pos, neg, pos_rate, num, size):
    positive_bags = []
    for _ in range(num):
        positive_instances = copy(pos)
        negative_instances = copy(neg)
        instances = []
        instances.append(pos.pop(
            randrange(len(pos))))

        for _ in range(size - 1):
            if random() < pos_rate:
                instance = positive_instances.pop(
                    randrange(len(positive_instances)))
            else:
                instance = negative_instances.pop(
                    randrange(len(negative_instances)))
            instances.append(instance)

        positive_bags.append(instances)
    return positive_bags


# %%
def create_negative_bags(neg, num, size):
    negative_bags = []
    for _ in range(num):
        negative_instances = copy(neg)
        instances = []
        for _ in range(size):
            instance = negative_instances.pop(
                randrange(len(negative_instances)))
            instances.append(instance)
        negative_bags.append(instances)

    return negative_bags


# %%
def read_file(path):
    file = open(path, "r")
    contents = " ".join(file.readlines())
    file.close()
    return contents


# %%
def tokenize_instances(instances, pool=None):
    if pool:
        read_docs = pool.map(read_file, instances)
        tokenized_docs = pool.map(preprocess_document, read_docs)
        docs = dict(zip(instances, tokenized_docs))

    else:
        docs = {path: preprocess_document(read_file(path))
                for path in instances}

    return docs

# %%


def make_bags(target, path='20_newsgroups', num_pos=50, num_neg=50, num_instances=50, pos_rate=0.03):
    path_dict = create_file_dict(path)

    positive_paths = path_dict[target]
    negative_paths = sum([path_dict[key]
                         for key in path_dict.keys() if key != target], [])

    positive_bags = create_positive_bags(
        pos=positive_paths, neg=negative_paths, num=num_pos, size=num_instances, pos_rate=pos_rate)
    negative_bags = create_negative_bags(
        neg=negative_paths, num=num_neg, size=num_instances)

    labels = [1 for _ in range(num_pos)] + [0 for _ in range(num_neg)]

    return positive_bags + negative_bags, labels


# %%
def tfidf(target, pool=None, num_pos=50, num_neg=50, num_instances=50, num_features=50, pos_rate=0.03):
    bags, labels = make_bags(target, num_pos=num_pos, num_neg=num_neg,
                             num_instances=num_instances, pos_rate=pos_rate)
    instances = sum([bag for bag in bags], [])

    docs = tokenize_instances(instances=instances, pool=pool)
    doc_bags = [[docs[instance] for instance in bag] for bag in bags]

    model = tfidf_model(num_features=num_features)
    model.create_dictionary(docs.values())
    model.initialize_model(docs.values())

    if pool:
        bow_bags = pool.map(model.create_features, doc_bags)
    else:
        bow_bags = [model.create_features(doc_bag) for doc_bag in doc_bags]

    return bow_bags, labels


# %%
def bow(target, pool=None, num_pos=50, num_neg=50, num_instances=50, num_features=50, pos_rate=0.03):
    bags, labels = make_bags(target, num_pos=num_pos, num_neg=num_neg,
                             num_instances=num_instances, pos_rate=pos_rate)
    instances = sum([bag for bag in bags], [])

    docs = tokenize_instances(instances=instances, pool=pool)
    doc_bags = [[docs[instance] for instance in bag] for bag in bags]

    model = bow_model(num_features=num_features)
    model.create_dictionary(docs.values())

    if pool:
        bow_bags = pool.map(model.create_features, doc_bags)
    else:
        bow_bags = [model.create_features(doc_bag) for doc_bag in doc_bags]

    return bow_bags, labels


# %%
def doc2vec(target, pool=None, num_pos=50, num_neg=50, num_instances=50, num_features=50, pos_rate=0.03, epochs=1):
    bags, labels = make_bags(target, num_pos=num_pos, num_neg=num_neg,
                             num_instances=num_instances, pos_rate=pos_rate)
    instances = sum([bag for bag in bags], [])

    docs = tokenize_instances(instances=instances, pool=pool)
    doc_bags = [[docs[instance] for instance in bag] for bag in bags]

    model = doc2vec_model(num_features=num_features)
    model.initialize_model(docs=docs.values())
    model.train_model(epochs=epochs)

    if pool:
        bow_bags = pool.map(model.create_features, doc_bags)
    else:
        bow_bags = [model.create_features(doc_bag) for doc_bag in doc_bags]

    return bow_bags, labels

from preprocess import preprocess_documents
from preprocess import preprocess_document
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class bow_model:
    def __init__(self, num_features=200):
        self.num_features = num_features
        self.dictionary = None

    def create_dictionary(self, docs):
        dictionary = Dictionary(docs)
        dictionary.filter_extremes(
            no_below=0, no_above=1, keep_n=self.num_features)
        self.dictionary = dictionary

    def create_features(self, docs):
        bow_corpus = [self.dictionary.doc2bow(
            doc, allow_update=False) for doc in docs]

        return [self.__convert_instance(i) for i in bow_corpus]

    def __convert_instance(self, instance):
        dict_instance = {x: 0 for x in range(self.num_features)}
        for i, v in instance:
            dict_instance[i] = v
        return list(dict_instance.values())


class tfidf_model:
    def __init__(self, num_features=200):
        self.num_features = num_features
        self.dictionary = None
        self.tfidf = None

    def create_dictionary(self, docs):
        dictionary = Dictionary(docs)
        dictionary.filter_extremes(
            no_below=0, no_above=1, keep_n=self.num_features)
        self.dictionary = dictionary

    def initialize_model(self, docs):
        bow_corpus = [self.dictionary.doc2bow(
            doc, allow_update=False) for doc in docs]
        self.tfidf = TfidfModel(bow_corpus)

    def create_features(self, docs):
        bow_corpus = [self.dictionary.doc2bow(
            doc, allow_update=False) for doc in docs]
        tfidf_corpus = self.tfidf[bow_corpus]
        return [self.__convert_instance(i) for i in tfidf_corpus]

    def __convert_instance(self, instance):
        dict_instance = {x: 0 for x in range(self.num_features)}
        for i, v in instance:
            dict_instance[i] = v
        return list(dict_instance.values())


class doc2vec_model:
    def __init__(self, num_features=200):
        self.num_features = num_features
        self.doc2vec = None
        self.tagged_docs = None

    def initialize_model(self, docs):
        self.doc2vec = Doc2Vec(
            vector_size=self.num_features)

        self.tagged_docs = [TaggedDocument(doc, [i])
                            for i, doc in enumerate(docs)]
        self.doc2vec.build_vocab(self.tagged_docs)

    def train_model(self, epochs=40):
        self.doc2vec.train(self.tagged_docs, total_examples=self.doc2vec.corpus_count,
                           epochs=epochs)

    def create_features(self, docs):
        return [self.doc2vec.infer_vector(doc) for doc in docs]

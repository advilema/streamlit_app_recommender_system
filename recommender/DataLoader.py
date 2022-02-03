import gensim
import pandas as pd
from stop_words import get_stop_words


class DataLoader:
    """
    Class used to load the data and make some preprocessing.
    The data is a CSV file, containing a certain number of articles.

    ...

    Attributes
    ----------
    model : gensim.models.keyedvectors.KeyedVectors
        NLP model used in the preprocessing
    path : str
        Path were the data are stored

    Methods
    -------
    load()
        Load the data from the path and drop nan values
    load_preprocess(columns=['leadtext'])
        Call load(), then preprocess the data saved in the columns and store the preprocessed data
        in the column data['preprocessed]
    _preprocess(doc)
        doc is a string. Tokenize doc, remove stop_words
    _tokenize(doc)
        doc in a string. Tokenize doc
    _remove_non_words(doc)
        remove words that are not in the model dictionary

    """

    def __init__(self, model=None, path='data/data.csv'):
        self.model = model
        if self.model is not None:
            assert type(model) is gensim.models.keyedvectors.KeyedVectors
        self.path = path

    def load(self):
        data = pd.read_csv(self.path)
        data = data.dropna()
        data.index = range(data.shape[0])
        return data

    def load_preprocess(self, columns=['leadtext']):  # possible columns are ['leadtext'], ['content'] or both together: ['leadtext', 'content']
        assert self.model is not None
        data = self.load()
        clean_article = []

        for i in range(data.shape[0]):
            doc = ''
            for column in columns:
                doc += data.iloc[i][column]
            doc = self._preprocess(doc)
            doc = self._remove_non_words(doc)
            clean_article.append(doc)

        data['preprocessed'] = clean_article

        return data

    def _preprocess(self, doc):
        letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
                   "u", "v", "w", "x", "y", "z"]
        other = ["wa", "ha", "one", "two", "id", "re", "http", "com", "mr", "image", "photo", "caption", "don", "sen",
                 "pic", "co",
                 "source", "watch", "play", "duration", "video", "momentjs", "getty", "images", "newsletter", "story",
                 "go", "like", "say",
                 "will", "just", "today", "can", "year", "make", "view", "read"]
        en_stop = get_stop_words('en')
        en_stop = en_stop[:174]
        en_stop += [word.capitalize() for word in en_stop]
        doc = doc.replace("\n", " ")
        doc = doc.replace("-", " ")
        doc = doc.replace("\'", "")

        doc = self._tokenize(doc)

        doc = [word for word in doc if not word in (en_stop + letters + other)]
        return doc

    def _tokenize(self, doc):
        tokens = [token for token in gensim.utils.tokenize(doc, lower=False, errors='ignore')
                  if 2 <= len(token) <= 15 and not token.startswith('_')]
        return tokens

    def _remove_non_words(self, doc):
        doc = [word for word in doc if self.model.has_index_for(word)]
        return doc

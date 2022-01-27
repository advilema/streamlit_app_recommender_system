import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import gensim



class Clustering:
    def __init__(self, data, model, cluster_by_words=True, column='preprocessed', n_clusters=10, n_init=100,
                 max_iter=1000):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter)
        try:
            type(model) is gensim.models.keyedvectors.KeyedVectors
        except ValueError:
            print('model must be a gensim.models.keyedvectors.KeyedVectors object')
        self.model = model
        if cluster_by_words:
            self.vec = self._make_words_vec(data[column])
        else:
            self.vec = self._make_articles_vec(data[column])

    def _make_articles_vec(self, documents):
        vec = []
        for doc in documents:
            n = len(doc)
            mean = 0
            for word in doc:
                mean += self.model.get_vector(word) / n
            vec.append(mean)
        return vec

    def _make_words_vec(self, documents):
        vec = []
        all_words = []
        for doc in documents:
            for word in doc:
                if word not in all_words:
                    all_words.append(word)
                    vec.append(self.model.get_vector(word))
        return vec

    def fit(self):
        self.kmeans.fit(self.vec)
        self.clusters = self.kmeans.cluster_centers_
        self.labels = self.kmeans.labels_

    def generate_topics(self):
        topics = []
        for i in range(self.n_clusters):
            topic = self.model.similar_by_vector(self.clusters[i], restrict_vocab=10000)[0][0]
            topics.append(topic)
        self.topics = topics
        return topics

    def plot_clusters(self, save_path=None):
        sc = StandardScaler()
        vec_transformed = sc.fit_transform(self.vec)

        pca = PCA(n_components=2)
        vec_projected = pca.fit_transform(vec_transformed)

        plt.figure(figsize=(15, 7))

        for label in range(self.n_clusters):
            points_cluster = np.transpose(vec_projected[np.where(self.labels == label)])
            plt.plot(points_cluster[0], points_cluster[1], '.', label=self.topics[label])
        plt.legend()
        if save_path is not None:
            plt.savefig(save_path)

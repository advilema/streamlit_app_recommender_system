import gensim
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import haversine as hs
import itertools


class Articles:
    '''
    This class serves only as a parent class. Use instead the three children classes below
    '''

    def __init__(self, data, topics, summary='leadtext', content='content', title='title'):
        self.topics = topics
        self.summary = np.array(data[summary])
        self.content = np.array(data[content])
        self.n_articles = len(self.summary)
        self.vec_by_topics = None
        self.titles = data[title]

    def save_vec_by_topics(self, path='data/vec_by_topics.npy'):
        np.save(path, np.array(self.vec_by_topics))
        return

    # return top articles by topic(s) of choice.
    # it is possible to combine many topics together: topic=['Bern', 'Mountain']

    def get_summary(self, idx):
        return self.summary[idx]

    def get_content(self, idx):
        return self.content[idx]

    def get_article(self, idx):
        return self.get_summary(idx) + self.get_content(idx)

    def get_title(self, idx):
        return self.titles[idx]

    def get_vec(self, idx):
        return self.vec_by_topics[idx]

    def plot_weights(self, idx, print_summary=True, print_content=False, save_path=None):
        if print_summary:
            print(self.summary[idx])
        if print_content:
            print(self.content[idx])

        plt.figure(figsize=[10, 5])
        plt.barh([i for i in range(len(self.topics))], self.vec_by_topics[idx])
        plt.yticks(ticks=np.arange(len(self.topics)), labels=self.topics)
        plt.show()
        if save_path is not None:
            plt.savefig(save_path)

    def _dist_articles_user(self, user):
        return norm(np.array(self.vec_by_topics) - np.array(user.vec), axis=1)

    def recommend_articles(self, user, n_recommendations=5):
        dist = self._dist_articles_user(user)
        n_selected = 0
        best_articles_ids = dist.argsort()
        recommended_articles_ids = []
        distances = []
        for idx in best_articles_ids:
            if idx not in user.seen_articles:
                recommended_articles_ids.append(idx)
                distances.append(dist[idx])
                n_selected += 1
                if n_selected == n_recommendations:
                    break
        recommended_articles_vec = []
        for idx in recommended_articles_ids:
            recommended_articles_vec.append(self.get_vec(idx))

        return np.array(recommended_articles_ids), np.array(recommended_articles_vec), np.array(distances)


class ArticlesVec(Articles):
    '''
    Use this class if you already have a vec_by_topics computed
    '''

    def __init__(self, data, topics, vec_by_topics, summary='leadtext', content='content'):
        super().__init__(data, topics, summary, content)
        assert vec_by_topics.shape[1] == len(topics)
        self.vec_by_topics = vec_by_topics


class ArticlesModel(Articles):
    '''
    Use this class if you don't have a vec_by_topics vector already computed (leave it as None), or if you
    want to make use of the method ids_top_articles_by_topic. To use this class you need to have model, a
    gensim.models.keyedvectors.KeyedVectors object
    '''

    def __init__(self, data, topics, model, vec_by_topics=None, summary='leadtext', content='content',
                 preprocessed='preprocessed',
                 power=4):
        super().__init__(data, topics, summary, content)
        self.preprocessed = data[preprocessed]
        try:
            type(model) is gensim.models.keyedvectors.KeyedVectors
        except ValueError:
            print('model must be a gensim.models.keyedvectors.KeyedVectors object')
        self.model = model
        if vec_by_topics is None:
            self.vec_by_topics = self._vec_by_topics(power)
        else:
            self.vec_by_topics = vec_by_topics

    def _vec_by_topics(self, power=4):
        vec_by_topics = []

        for doc in self.preprocessed:
            vec_by_topic = [0 for topic in self.topics]
            for i, topic in enumerate(self.topics):
                for word in doc:
                    vec_by_topic[i] += self.model.similarity(topic, word) ** power
            tot = np.sum(vec_by_topic)
            vec_by_topic = [vec / tot for vec in vec_by_topic]  # normalize
            vec_by_topics.append(vec_by_topic)

        return vec_by_topics

    def get_preprocessed(self, idx):
        return self.preprocessed[idx]

    def ids_top_articles_by_topic(self, topic, top_n=10):
        scores = [[0, i] for i in range(self.n_articles)]

        for i, text in enumerate(self.preprocessed):
            scores[i][0] = self.model.n_similarity(text, topic)

        scores.sort(key=lambda x: x[0], reverse=True)
        selected = np.array(scores[:top_n])
        return selected.transpose()[1]


class ArticlesGeo(ArticlesVec):
    def __init__(self, data, topics, vec_by_topics, summary='leadtext', content='content'):
        super().__init__(data, topics, vec_by_topics, summary=summary, content=content)
        self.coordinates = np.concatenate((np.array([data['lat']]), np.array([data['lng']])), axis=0).transpose()

    def get_coordinates(self, idx):
        return self.coordinates[idx].tolist()

    def get_geo_recommendations(self, user, max_dist=20, max_n_activities=3, n_recommendations=5, n_best_articles=30,
                                remove_rep=True):
        recommended_articles_ids, recommended_articles_vec, _ = self.recommend_articles(user, n_best_articles)
        coordinates = []
        for idx in recommended_articles_ids:
            coordinates.append(self.get_coordinates(idx))

        connection_dict = {}
        for idx in recommended_articles_ids:
            connection_dict[idx] = []
        for i, idx in enumerate(recommended_articles_ids):
            for j, idx_2 in enumerate(recommended_articles_ids[i + 1:]):
                if hs.haversine(coordinates[i], coordinates[j + i + 1]) < max_dist:
                    connection_dict[idx].append(idx_2)
                    connection_dict[idx_2].append(idx)

        combinations_ids = []
        for idx in connection_dict:
            combinations = self._n_combinations(connection_dict[idx], max_n_activities)
            [elem.append(idx) for elem in combinations]
            combinations.append([idx])
            combinations_ids.extend(combinations)

        best_dists, best_ids, best_combs, best_combs_vec = self._get_best(
            np.array(combinations_ids, dtype=object), user, n_recommendations, remove_rep)

        best_combs_coor = []
        for comb in best_combs:
            coor = []
            for idx in comb:
                coor.append(self.get_coordinates(idx))
            best_combs_coor.append(coor)

        return best_combs, best_combs_vec, best_dists, best_combs_coor

    @staticmethod
    def _n_combinations(array, n):
        max_n = min(len(array), n)
        m = 1
        combinations = []
        while m < max_n:
            iter_comb = itertools.combinations(array, m)
            comb = [list(elem) for elem in iter_comb]
            combinations.extend(comb)
            m += 1
        return combinations

    def _get_comb_vec(self, combinations):
        comb_vec = []
        for comb in combinations:
            vec = np.zeros(len(self.topics))
            for elem in comb:
                vec += self.get_vec(elem)
            vec /= len(comb)
            comb_vec.append(vec.tolist())
        return comb_vec

    def _get_best(self, combinations_ids, user, n_best, remove_rep):
        vec = self._get_comb_vec(combinations_ids)
        dist = np.linalg.norm(user.get_vec() - vec, axis=1)
        # remove repeated combinations
        for i, d in enumerate(dist):
            if d == 1:
                continue
            for j, d_2 in enumerate(dist[i + 1:]):
                if d == d_2:
                    dist[j + i + 1] = 1
        best_ids = np.argsort(dist)
        best_dist = dist[best_ids]
        best_comb_ids = combinations_ids[best_ids]
        best_comb_vec = np.array(vec)[best_ids]
        if remove_rep:
            seen = set()
            for i, comb in enumerate(best_comb_ids):
                comb_set = set(comb)
                if seen.intersection(comb_set):
                    best_dist[i] = 1
                seen = seen.union(set(comb))
            best_ids_no_rep = np.argsort(best_dist)
            best_ids = best_ids[best_ids_no_rep]
            best_dist = best_dist[best_ids_no_rep]
            best_comb_ids = best_comb_ids[best_ids_no_rep]
            best_comb_vec = best_comb_vec[best_ids_no_rep]

        idx_dist_1 = len(best_dist)
        for i, dist in enumerate(best_dist):
            if dist == 1:
                idx_dist_1 = i
                break
        n_best = min(n_best, idx_dist_1)

        return best_dist[:n_best], best_ids[:n_best], best_comb_ids[:n_best], best_comb_vec[:n_best]



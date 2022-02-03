import gensim
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import haversine as hs
import itertools


class Articles:
    """
    This class serves only as a parent class, and shouldn't be used directly.
    Used instead one of the child classes below depending on your needs.

    ...

    Attributes
    ----------
    topics : list(str)
        The possible topics to describe the vacation type
    summary : list(str)
        Each entry in the list represent a summary of an article
    content : list(str)
        Each entry in the list represent the content of an article
    title : list(str)
        Each entry in the list represent the title of an article
    n_articles : int
        The number of articles
    vec_by_topics : list(list(floats))
        Each entry in the list represent a topic based vector representation of the article

    Methods
    -------
    save_vec_by_topics(path='data/vec_by_topics.npy')
        Save vec_by_topics in path
    get_summary(idx)
        Return summary[idx]
    get_content(idx)
        Return content[idx]
    get_article(idx)
        Return summary[idx] + content[idx]
    get_title(idx)
        Return title[idx]
    get_vec(idx)
        Return vec_by_topics[idx]
    plot_weights(idx, print_summary=True, print_content=False, save_path=None)
        Plot the vec_by_topics[idx] in a horizontal bar chart and save it in save_path if is not None
        If print_summary: print(summary[idx])
        if print_content: print(content[idx])
    _dist_articles_user(user)
        Return euclidean distance between an article and the user
    recommend_articles(user, n_recommendations=5)
        Recommend to the user the best n_recommendations articles that are the nearest in the euclidean
        distance to the user and the the user has not seen yet

    """

    def __init__(self, data, topics, summary='leadtext', content='content', title='title'):
        self.topics = topics
        self.summary = np.array(data[summary])
        self.content = np.array(data[content])
        self.title = data[title]
        self.n_articles = len(self.summary)
        self.vec_by_topics = None

    def save_vec_by_topics(self, path='data/vec_by_topics.npy'):
        np.save(path, np.array(self.vec_by_topics))
        return

    def get_summary(self, idx):
        return self.summary[idx]

    def get_content(self, idx):
        return self.content[idx]

    def get_article(self, idx):
        return self.get_summary(idx) + self.get_content(idx)

    def get_title(self, idx):
        return self.title[idx]

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
    """
    Use this class if you already have a vec_by_topics computed

    Attributes
    ----------
    All of the super() Attributes


    Methods
    -------
    All of the super() Methods
    """

    def __init__(self, data, topics, vec_by_topics, summary='leadtext', content='content'):
        super().__init__(data, topics, summary, content)
        assert vec_by_topics.shape[1] == len(topics)
        self.vec_by_topics = vec_by_topics


class ArticlesModel(Articles):
    """
    Use this class if you don't have a vec_by_topics vector already computed (leave it as None), or if you
    want to make use of the method ids_top_articles_by_topic. To use this class you need to have model, a
    gensim.models.keyedvectors.KeyedVectors object

    Attributes
    ----------
    All of the super() Attributes, plus:

    preprocessed : list(list(str))
        Each entry in the list is a list of string which is the preprocessed version of the content of the article
    model : gensim.models.keyedvectors.KeyedVectors
        The NLP model that we are going to use to do the embedding of the words into the article

    Methods
    -------
    All of the super() Methods, plus:

    _vec_by_topics(power=4)
        convert compute vec_by_topics using preprocessed and model. power is an hyperparameter,that regulate a
        softmax behaviour: if it's high the vector representing the topics will have an high weight in very few topics,
        while if it's small, the weight is going to be distributed more uniformly among many topics.
    get_preprocessed(idx)
        Return preprocessed[idx]
    ids_top_articles_by_topic(topic, top_n=10)
        topic can be any word or list of words (e.g. [Bern, mountain])
        Return the top_n most similar articles to topic
    """

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
    """
    This class generalize ArticlesVec, allowing to get recommendation for the user based also on the location of the
    attractions that are described in the articles.

    Attributes
    ----------
    All of the super() Attributes, plus:

    coordinates : list(list(floats))
        Each entry in the list is a list of two elements: the latitude and the longitude of the location of the
        attractions described in the article


    Methods
    -------
    All of the super() Methods, plus:

    get_coordinates(idx)
        Return coordinates[idx]
    get_geo_recommendations(user, max_dist=20, max_n_activities=3, n_recommendations=5, n_best_articles=30,
                                remove_rep=True)
        Return a n_recommendations combinations of activities (articles) for the user, each combination having a maximum
        of max_n_activities articles, each articles within a combination have a distance with the others of less than
        max_dist.
        See help(ArticlesGeo.get_geo_recommendations) for a more detailed description or below in the method description
    _n_combinations(array,n)
        Get all of the combinations of the possible combination of the elements in the array, such that at most you
        combine together n elements of the array
    _get_comb_vec(combinations)
        Get the topics based vector representation of the combinations of articles in "combinations", by doing the mean
        of all of the articles in the combination
    _get_best(combinations_ids, user, n_best, remove_rep)
        Select the n_best combinations for the user from combinations_ids.
        If remove_rep: don't allow the same article to be in more than one combination
    """

    def __init__(self, data, topics, vec_by_topics, summary='leadtext', content='content'):
        super().__init__(data, topics, vec_by_topics, summary=summary, content=content)
        self.coordinates = np.concatenate((np.array([data['lat']]), np.array([data['lng']])), axis=0).transpose()

    def get_coordinates(self, idx):
        return self.coordinates[idx].tolist()

    def get_geo_recommendations(self, user, max_dist=20, max_n_activities=3, n_recommendations=5, n_best_articles=80,
                                remove_rep=True):
        """
        Return the best n_recommendations combinations of activities for the user, such that:
            - the articles in the combinations have not been seen by the user yet
            - each combination is a list of a maximum max_n_activities articles, and the distance between the location
            of the activities described by the articles is less than max_dist kilometers
            - if remove_rep the same article is not allowed to be in more than one combination

        Parameters
        ----------
        user : User
            The user we are generating the recommendations for
        max_dist : float
            Maximum distance between activities in a combination expressed in kilometers
        max_n_activities : int
            Maximum number of articles (activities) in a combination
        n_recommendations : int
            Number of combinations that will be returned
        n_best_articles : int
            To create the combinations only the best n_best_articles for the user will be considered. The higher this
            number, the slower and the more accurate the algorithm. Usually 80 should be a good tradeoff between speed
            and accuracy
        remove_rep : bool
            If True, the same article is not allowed to be in more than one combination



        Returns
        -------
        best_combs : np.array(list(list(int)))
            Each entry is a list of int representing the ids of the articles into that combination.
            It's an ordered vector: the first combination is the best and the last one the worst.
            len(best_combs) <= n_recommendations
        best_combs_vec : np.array(list(list(float)))
            The vector representations of the combinations described above
        best_dists : np.array(list(float))
            The distances between the vector representation of the combinations described above and the user
        best_combs_coor : np.array(list(list(list(float))))
            Each entry is a list that represent a combination. Each element in that list is itself a list that
            represent the latitude and longitude of the position of the attraction described by the articles in
            the combination
        """

        recommended_articles_ids, recommended_articles_vec, _ = self.recommend_articles(user, n_best_articles)
        coordinates = []
        for idx in recommended_articles_ids:
            coordinates.append(self.get_coordinates(idx))

        # compute the graph where two articles are linked if their positions are close enough within each others
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



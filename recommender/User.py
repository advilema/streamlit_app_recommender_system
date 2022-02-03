import numpy as np
import matplotlib.pyplot as plt


class User:
    """
    A class used to represent the User

    ...

    Attributes
    ----------
    topics : list(str)
        The possible topics to describe the vacation type
    choices : list(str)
        The specific topics the user is interested in
    vec : list(floats)
        A vector representation of the user. The vector is normalized and each
        weight is proportional to the interest of the user for a specific topic
    seen_articles : list(int)
        The indices of the articles that the user has already seen
    learning_rate : float
        Number between 0 and 1 that indicates how much the vector representation of the
        user is changed by seeing new articles

    Methods
    -------
    _make_vec()
        Given choices and topics create vec
    reset()
        Reset the vec to its original state (by calling _make_vec())
    get_seen_articles()
        Return seen_articles
    get_vec()
        Return vec
    update(articles)
        Given a list of articles (represented as vector with length len(topics)) update vec
    add_seen_articles(articles_ids)
        Append to seen_articles new articles ids
    plot_weights(save_path=None)
        Plot vec in a horizontal bar char
    """
    def __init__(self, topics, choices, learning_rate=0.1):  # 0<learning_rate<=1
        self.topics = topics
        self.choices = choices
        self.vec = self._make_vec()
        self.seen_articles = []
        self.learning_rate = learning_rate

    def _make_vec(self):
        vec = np.array([1 if topic in self.choices else 0 for topic in self.topics])
        vec = vec / sum(vec)
        return vec

    def reset(self):
        self.vec = self._make_vec()
        self.seen_articles = []

    def get_seen_articles(self):
        return self.seen_articles

    def get_vec(self):
        return self.vec

    def update(self, articles):
        for article in articles:
            self.vec = ((1 - self.learning_rate) * self.vec + self.learning_rate * np.array(article))
        return

    def add_seen_articles(self, article_ids):
        self.seen_articles.extend(article_ids)
        return

    def plot_weights(self, save_path=None):
        fig, ax = plt.subplots()
        fig.set_figheight(10)
        fig.set_figwidth(5)
        ax.barh([i for i in range(len(self.topics))], self.vec)
        plt.yticks(ticks=np.arange(len(self.topics)), labels=self.topics)
        if save_path is not None:
            plt.savefig(save_path)
        return fig

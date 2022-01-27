import numpy as np
import matplotlib.pyplot as plt

class User:
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

from recommender.Articles import ArticlesModel
from recommender.DataLoader import DataLoader
import argparse
import os
from gensim.models import KeyedVectors
import numpy as np
import gensim.downloader as api

def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_vec', default='data/vec_articles.npy',
                        help='The path in which to store the vector representation of the articles dependent on the topics')
    parser.add_argument('--path_topics', default='data/topics.npy',
                        help='The path where to store the topics')
    parser.add_argument('--model_name', default='word2vec-google-news-300',
                        help='The path where the model is stored')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print('Initialization started, it may takes a while, especially the first time...')
    topics = ['city', 'children', 'museums', 'art', 'music', 'traditions', 'luxury', 'nature', 'mountains',
              'relax', 'lake', 'wine', 'food', 'bike', 'ski', 'hiking', 'spa', 'summer', 'winter']
    topics.sort()

    args = cli()

    np.save(args.path_topics, np.array(topics))

    print('loading the nlp model...')
    if not os.path.isfile(api.load(args.model_name, return_path=True)):
        model = api.load(args.model_name)
    else:
        model = KeyedVectors.load_word2vec_format(api.load(args.model_name, return_path=True), binary=True, limit=500000)
    print('nlp model successfully loaded!')

    loader = DataLoader(model=model)
    data = loader.load_preprocess()

    articles = ArticlesModel(data, topics, model)
    articles.save_vec_by_topics(args.path_vec)

    print('Initialization successful!')

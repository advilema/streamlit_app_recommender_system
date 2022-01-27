import streamlit as st
import numpy as np
from recommender.DataLoader import DataLoader
from recommender.User import User
from recommender.Articles import ArticlesVec


if __name__ == '__main__':
    n_recommendations = 5
    learning_rate = 1

    loader = DataLoader()
    data = loader.load()
    vec_articles = np.load('data/vec_articles.npy')
    topics = np.load('data/topics.npy')

    articles = ArticlesVec(data, topics, vec_articles)

    st.title('Recommendation System')
    choices = st.multiselect('What kind of vacation are you looking for?', topics)

    if 'selection_done' not in st.session_state:
        st.session_state.selection_done = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'seen_articles_ids' not in st.session_state:
        st.session_state.seen_articles_ids = []

    def update_done():
        st.session_state.selection_done = not st.session_state.selection_done
        st.session_state.user = User(topics, choices, learning_rate)


    st.button('Done', key='done1', on_click=update_done)

    if st.session_state.selection_done:

        recommended_articles_ids, recommended_articles_vec, distances = articles.recommend_articles(st.session_state.user, n_recommendations)
        articles_chosen = []
        for i in range(n_recommendations):
            articles_chosen.append(st.checkbox(articles.get_summary(recommended_articles_ids[i])))
            st.write('Similarity score: {0:2.1f}%'.format(100*(1-distances[i])))
            st.write()
        selected_articles = st.button('Done', key='done2')
        if selected_articles:
            articles_chosen_ids = np.where(articles_chosen)[0]
            st.session_state.user.add_seen_articles(recommended_articles_ids[articles_chosen_ids].tolist())
            articles_chosen_vec = recommended_articles_vec[articles_chosen_ids]
            st.session_state.user.update(articles_chosen_vec)
            show_results = st.button('Update')

import streamlit as st
import numpy as np
import folium
from streamlit_folium import folium_static
from recommender.DataLoader import DataLoader
from recommender.User import User
from recommender.Articles import ArticlesGeo


CH_COOR = [46.8, 8.8]  # coordinates of the center of Switzerland
COLORS = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
          'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
          'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']


def make_recommendations():
    with st.spinner('Wait for it...'):
        best_combs_ids, best_combs_vec, best_dists, best_combs_coor = articles.get_geo_recommendations(
            st.session_state.user,
            max_dist=max_dist,
            max_n_activities=max_n_activities,
            n_recommendations=n_recommendations,
            n_best_articles=n_best_articles
        )
    st.session_state.best_combs_ids = best_combs_ids
    st.session_state.best_combs_vec = best_combs_vec
    st.session_state.best_dists = best_dists
    st.session_state.best_combs_coor = best_combs_coor
    return

def update_done():
    st.session_state.selection_done = not st.session_state.selection_done
    if st.session_state.selection_done:
        st.session_state.user = User(topics, choices, learning_rate)
        make_recommendations()
        return


if __name__ == '__main__':
    n_recommendations = 5
    n_best_articles = 100 #lower to speed up. The higher the most accurate the results but the slowest the app
    learning_rate = 1

    loader = DataLoader()
    data = loader.load()
    vec_articles = np.load('data/vec_articles.npy')
    topics = np.load('data/topics.npy')

    articles = ArticlesGeo(data, topics, vec_articles)

    st.title('Your swiss day-travel planner :snowboarder: :sunglasses:')
    choices = st.multiselect('What kind of vacation are you looking for?', topics)

    col_activities, col_km = st.columns(2)
    with col_activities:
        max_n_activities = st.slider('Number of activities', 1, 3, 1)
    with col_km:
        max_dist = st.slider('Maximum distance in km between activities', 5, 40)

    if 'selection_done' not in st.session_state:
        st.session_state.selection_done = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'seen_articles_ids' not in st.session_state:
        st.session_state.seen_articles_ids = []

    col_done, col_user = st.columns((1,1))
    with col_done:
        button_text = 'Done' if not st.session_state.selection_done else 'Clear'
        st.button(button_text, key='done1', on_click=update_done)
    with col_user:
        show_user = st.checkbox('show user vector', value=False)

    if st.session_state.selection_done:
        if show_user:
            sidebar = st.sidebar.container()
            with sidebar:
                plot = st.session_state.user.plot_weights()
                st.write(plot)
        best_combs_ids = st.session_state.best_combs_ids
        best_combs_vec = st.session_state.best_combs_vec
        best_dists = st.session_state.best_dists
        best_combs_coor = st.session_state.best_combs_coor

        combs_chosen = []
        for i, comb, dist in zip(range(len(best_combs_ids)),best_combs_ids, best_dists):
            col_button, col_text = st.columns((1,20))
            with col_button:
                combs_chosen.append(st.checkbox('', key=comb))
            with col_text:
                color = '<span style="color: '+COLORS[i%len(COLORS)]+'"> ⬤ </span>'
                score = '- Similarity score: {0:2.1f}%'.format(100 * (1 - dist))
                #st.markdown('<span style="color: '+COLORS[i%len(COLORS)]+'"> o </span>', unsafe_allow_html=True)
                st.write(color+score, unsafe_allow_html=True)
                for idx in comb:
                    #article_description = '**'+articles.get_title(idx)+'**'
                    #article_description += '  \n  ' + articles.get_summary(idx)
                    #st.write(article_description)
                    with st.expander(articles.get_title(idx)):
                        st.write(articles.get_summary(idx))

        map = folium.Map(location=CH_COOR, zoom_start=7,
                             tiles='cartodbpositron', width=800, height=400)
        _ = [[folium.CircleMarker(location=c, radius=1,
                             color=COLORS[i%len(COLORS)]).add_to(map)
         for c in coor] for i, coor in enumerate(best_combs_coor)]
        folium_static(map)

        has_selected_articles = st.button('Done', key='done2')

        if has_selected_articles:
            combs_chosen_ids = np.where(combs_chosen)[0]
            articles_chosen = []
            _ = [articles_chosen.extend(comb) for comb in best_combs_ids[combs_chosen_ids]]
            st.session_state.user.add_seen_articles(articles_chosen)
            articles_chosen_vec = best_combs_vec[combs_chosen_ids]
            st.session_state.user.update(articles_chosen_vec)
            show_results = st.button('Update', on_click=make_recommendations())

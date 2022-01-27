FROM python:3.8.3
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
EXPOSE 8501
COPY . /app
COPY data/data.csv /app/data/data.csv
COPY data/topics.npy /app/data/topics.npy
COPY data/vec_articles.npy /app/data/vec_articles.npy
COPY data/vec_by_topics.npy /app/data/vec_by_topics.npy
ENTRYPOINT ["streamlit", "run"]
CMD ["geoapp.py"]
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def display_topics(model, feature_names, num_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-num_top_words - 1:-1]]))

def display_documents_by_topic(documents, titles, tf_topics, topic_idx):
    top_docs = list(documents[np.argsort(tf_topics[:, topic_idx])[::-1]][:10])
    top_titles = list(titles[np.argsort(tf_topics[:, topic_idx])[::-1]][:10])
    for title, doc in zip(top_titles, top_docs):
        print(title)
        print(doc)
        print()

def display_closest_articles(documents, titles, tf_topics, article_name, func='cosine'):
    article_topic_probs = tf_topics[titles == article_name]
    article_idx = titles[titles == article_name].index[0]
    if func == 'cosine':
        similarities = cosine_similarity(article_topic_probs, tf_topics)[0]
    elif func == 'euclidean':
        similarities = -1 * euclidean_distances(article_topic_probs, tf_topics)[0]
    sorted_similarities_idxs = np.argsort(similarities)[::-1]
    # Remove our article idx
    sorted_similarities_idxs = sorted_similarities_idxs[sorted_similarities_idxs != article_idx]
    sorted_similarities = similarities[sorted_similarities_idxs]
    most_related_titles = titles[sorted_similarities_idxs]

    for title, sim in zip(most_related_titles[:10], sorted_similarities[:10]):
        print(title, "--", round(sim, 2))

def example_lda_flow(filepath):
    """This example uses a "spooky_wikipedia.csv" data file."""
    df = pd.read_csv(filepath)

    # df = clean_df(df)

    # Sample 1000 rows from the data (if desired)
    #df_sample = df.sample(1000, random_state=42)
    #df_sample.sort_index(inplace=True)
    df_sample = df

    # Count-vectorize data
    vectorizer = CountVectorizer(max_df=0.85, min_df=2, max_features=1000, stop_words='english')
    tf = vectorizer.fit_transform(df_sample['text'])
    tf_feature_names = vectorizer.get_feature_names()
    documents = df_sample['text'].reset_index(drop=True)
    titles = df_sample['title'].reset_index(drop=True)

    # Create LDA instance
    # Create an LDA instance and think about what each of the parameters mean.
    # In our use case, what does n_components represent?
    # How do we input our alpha and beta priors?
    # Use the 'online' learning method and n_jobs=-1 (all cores) or -2
    # (all cores but one) to speed up your processing.
    lda = LatentDirichletAllocation(n_components=10, learning_method='online', n_jobs=-1, random_state=42)
    lda.fit(tf)

    # Explore the topics
    num_top_words = 10
    display_topics(lda, tf_feature_names, num_top_words)

    # Transform term-frequency matrix in order to get topics for each article
    tf_topics = lda.transform(tf)
    display_documents_by_topic(documents, titles, tf_topics, 0)

    # Now, let's use a function that gets the closest articles to a query article
    another = 'y'
    while another == 'y':
        article_name = input('Find similar spooky Wiki articles: ')
        distance_func = input('What distance func do you want to use? ')
        display_closest_articles(documents, titles, tf_topics, article_name, distance_func)
        another = input('Want to find another? (y/n) ')

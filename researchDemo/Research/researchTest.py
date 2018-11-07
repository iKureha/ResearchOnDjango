# -*- encoding: utf-8 -*-
import sys
sys.path.append("/Users/wang 1/researchDemo/researchDemo/Research/")

import twitterAPIKey
import stop_words
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import nltk.stem


from sklearn.model_selection import GridSearchCV
# import mglearn
# from sklearn.pipeline import make_pipeline
# from sklearn.linear_model import LogisticRegression
stop = stop_words.stop_words


###################################################
# Link to Twitter
try:
    api = twitterAPIKey.linkToTwitter()
except RuntimeError:
    print('Fail to access to Twitter!\n')
else:
    print("Access to Twitter successfully!\n")


def number_of_following(user_name):
    twitter_name = user_name
    following_list = api.friends_ids(screen_name=twitter_name)
    return user_name + " has followed " + str(len(following_list)) + " accounts."


###################################################
# Get tweets from users' following list
class TimeLineContent:

    def __init__(self, id):
        self.id = id
        self.name = api.get_user(id=self.id).name

    def get_timeline(self, count):
        print("\nGetting " + self.name + "'s tweets...")
        try:
            timelines = api.user_timeline(id=self.id, count=count)
        except:
            print("Failed to get " + self.name + "'s tweets!")
        else:
            print("Got " + self.name + "'s tweets successfully!")
            return timelines

    def output_timeline(self, count=500):
        timelines = self.get_timeline(count)
        timeline_new = ""
        try:
            for timeline in timelines:
                timeline_new += timeline.text.replace('\n', ' ')
        except:
            pass
        else:
            return timeline_new


###################################################
# delete not English words
def englishlize(textList):
    temp = []
    for i in textList:
        temp2 = []
        for n in i.split(" "):
            if re.match("^[@#]*[A-Za-z0-9]+$", n):
                temp2.append(n)
        temp.append(" ".join(temp2))
    return temp


###################################################
# Get favorite contents
def favorites(account_id):
    favorites_list = []
    for i in api.favorites(id=account_id):
        vect = CountVectorizer(max_features=150, stop_words=stop)
        i = englishlize(i.text.replace('\n', ' ').split(' '))
        vect.fit_transform(i)
        favorites_list.append(" ".join(vect.get_feature_names()))
    print(favorites_list)
    return favorites_list


###################################################
# LDA model
def lda_clustering(new_timeline, n=5):
    print('\nStart to do clustering . . .')
    #################
    # Bag of Words
    # Delete stop words
    # Delete non-english words
    # parameters = {'learning_method': ['batch', 'online'],
    #               'n_topics': [5, 10, 15, 20, 25],
    #               'perp_tol': [0.001, 0.01, 0.1],
    #               'doc_topic_prior': [0.001, 0.01, 0.05, 0.1, 0.2],
    #               'topic_word_prior': [0.001, 0.01, 0.05, 0.1, 0.2],
    #               'max_iter': [1000]}
    vect = CountVectorizer(max_features=40000, min_df=.1, stop_words=stop)
    temp = englishlize(new_timeline)
    X = vect.fit_transform(temp)
    #################
    # LDA
    lda = LatentDirichletAllocation(n_topics=n, learning_method='batch', max_iter=25, random_state=0)
    lda.fit_transform(X)
    # lda = LatentDirichletAllocation()
    # model = GridSearchCV(lda, parameters)
    # model.fit(X)

    # print(model.best_estimator_)
    # print(model.best_params_)

    feature_names = np.array(vect.get_feature_names())

    topic_list = []
    index = 0
    for topic_idx, topic in enumerate(lda.components_):
        # message = "Topic #%d: " % topic_idx
        message = ""
        message += " ".join([feature_names[i] for i in topic.argsort()[:-10 - 1:-1]])
        topic_list.append(message)
        print('\n', "#Topic", index, ":", message)
        index += 1
    print('\nFinishing clustering . . .')
    return topic_list


###################################################
# cosine similarity
def cos_similarity(key_word_list, profile):
    train_set = [key_word_list]
    train_set.append(" ".join([x for x in profile]))
    count_vectorizer = CountVectorizer()
    count_matrix_train = count_vectorizer.fit_transform(train_set)
    similarity = cosine_similarity(count_matrix_train[0:1], count_matrix_train)
    return similarity


###################################################
# main
def runLDA(user_name):
    new_timeline = []
    following_list = api.friends_ids(screen_name=user_name)
    print(len(following_list), "in following list.")
    following_count = 0

    for i in following_list:
        following_count += 1
        try:
            temp = TimeLineContent(i)
        except:
            print('\nFailed to get information from user id:', i)
            continue
        else:
            new_timeline.append(str(temp.output_timeline(200)))
        print("(", following_count, "/", len(following_list), ") Done")
        # if following_count == 3:
        #     break

    topic_list = lda_clustering(new_timeline)
    return topic_list



def runCos(user_name, topic_list):
    topic_index = 0
    favorites_list = favorites(user_name)
    result = []
    for i in topic_list:
        print('\nTopic #', topic_index, '***************')
        result.append("Cosine similarity is " + str(cos_similarity(i, favorites_list)[0][1]))
        print("Cosine similarity is", cos_similarity(i, favorites_list)[0][1])
        topic_index += 1
    return result


def local_test():
    f = open("/Users/wang 1/researchDemo/researchDemo/Research/timeline.txt", "r+")
    text = f.read()
    timeline = text[2:-2].split("', '")
    f.close()
    return timeline



# twitter_name = input("アカウントを入力してください：")
# twitter_name = "MyTopicTest"
# new_timeline = []
# following_list = api.friends_ids(screen_name=twitter_name)
# print(len(followingList), "in following list.")
# followingCount = 0
#
# for i in following_list:
#     following_count += 1
#     try:
#         temp = TimeLineContent(i)
#     except:
#         print('\nFailed to get information from user id:', i)
#         continue
#     else:
#         new_timeline.append(str(temp.output_timeline(200)))
#     print("(", following_count, "/", len(following_list), ") Done")
#     # if following_count == 3:
#     #     break


# f = open('timeline.txt', "r+")
# f.write(str(new_timeline))
# text = f.read()
# timeline = text[2:-2].split("', '")
#
# lda_clustering(timeline)
#
# f.close()
# topic_list = lda_clustering(new_timeline, 10)

# print('\nFavorites:')
# favorites_list = favorites(twitter_name)
# print()

# topic_index = 0
# for i in topic_list:
#     print('\nTopic #', topic_index, '***************')
#     print("Cosine similarity is", cos_similarity(i, favorites_list)[0][1])
#     topic_index += 1


# a = api.get_user(id='PokemonGoApp')
# print('\nPokemonGpApp: ' + a.description)






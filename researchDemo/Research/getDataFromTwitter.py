# coding: utf-8

import tweepy
# import _mysql
import sys
import twitterAPIKey as tk

reload(sys)
sys.setdefaultencoding('utf8')


# Link to twitterAPI
auth = tweepy.OAuthHandler(tk.getConsumerKey(), tk.getConsumerSecret())
auth.set_access_token(tk.getAccessToken(), tk.getAccessTokenSecret())
api = tweepy.API(auth)


# Output progress
def progress_count(progress, total):
    sys.stdout.write('\rfinishing...{0}'.format(progress))
    sys.stdout.flush()


# Connect to MySQL
db = tk.getConnectToDB()

# fresh status
# api.update_status('hello world!')

# get timeline
# public_tweets = api.home_timeline(count=100)
# i = 0
# for tweet in public_tweets:
#     print i, '.' , tweet.text
#     print;print
#     i += 1


# know user id, output user name
me = api.get_user(screen_name="RoxasWy")
# print me.screen_name


# put following list to MySQL
f_train = open('/Users/wangyu/PycharmProjects/twitterTest/data/trainingData', 'a')
f_test = open('/Users/wangyu/PycharmProjects/twitterTest/data/testData', 'a')
a = api.user_timeline(id='BBC', count=144)
for tweets in a :
    b = str(tweets).replace("\n", "")
    f_test.write(b + "\n")
f_test.close()

num = 0
for following in api.friends_ids(screen_name="BBC"):
    user = api.get_user(following)
    timeline = api.user_timeline(id=following, count=100)

    # dict_tweet_id = {}
    # dict_tweet_text = {}
    # i = 0
    for tweets in timeline:
        # dict_tweet_id[i] = tweets.id
        # dict_tweet_text[i] = tweets.text.replace('"', "'")
        # i += 1
        a = str(tweets.text)
        b = a.replace("\n"," ")
        f_train.write(b)

    # print dict_tweet_id
    # print num, user.screen_name, user.id, ": ", dict_tweet_id[0], dict_tweet_text[0], dict_tweet_id[1], dict_tweet_text[1], dict_tweet_id[2], dict_tweet_text[2]
    # store into mysql
    # db.query("insert into tweets_info values(" + str(me.id) + ","  + str(user.id) + ",'" + user.screen_name + "'," + str(dict_tweet_id[0]) + ',"' + dict_tweet_text[0] + '",' + str(dict_tweet_id[1]) + ',"' + dict_tweet_text[1] + '",' + str(dict_tweet_id[2]) + ',"' + dict_tweet_text[2] + '")')
    progress_count(num, 100)
    f_train.write("\n")
    num += 1

f_train.close()
db.close()
print;print "OK!"




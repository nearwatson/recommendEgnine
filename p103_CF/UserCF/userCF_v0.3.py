import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import operator

print("Data Import Starting...")

moviesPath = "/Users/Keson/Desktop/test_data/movies.csv"
ratingsPath = "/Users/Keson/Desktop/test_data/ratings.csv"
userinfosPath = "/Users/Keson/Desktop/test_data/user_info.csv"
moviesDF = pd.read_csv(moviesPath, index_col = None)
ratingsDF = pd.read_csv(ratingsPath, index_col = None)
userinfoDF = pd.read_csv(userinfosPath, index_col = None)

print("Import data finished.")

trainRatingsPivotDF = pd.pivot_table(ratingsDF[['userId', 'movieId', 'rating']], columns=['movieId'],
                                            index=['userId'], values='rating', fill_value=0)
moviesMap = dict(enumerate(list(trainRatingsPivotDF.columns)))
usersMap = dict(enumerate(list(trainRatingsPivotDF.index)))
ratingValues = trainRatingsPivotDF.values.tolist()


print(ratingValues)
#moviesMap : {0:1,1:2,3:3...}
#usersMap: {0:1,1:2,3:3...}
#ratingValues: 矩阵变成list 每一行变成list的一个值!   用户对每一个电影打的分，没有就是0.0 [0.0, 0.0, 0.0, 1.5...]


#利用余弦相似度计算用户之间的相似度
def calCosineSimilarity(list1, list2):
    res = 0
    denominator1 = 0
    denominator2 = 0
    for (val1, val2) in zip(list1, list2):
        res += (val1 * val2)
        denominator1 += val1 ** 2
        denominator2 += val2 ** 2
    return res / (math.sqrt(denominator1 * denominator2))

## 根据用户对电影的评分，来判断每个用户间相似度
userSimMatrix = np.zeros((len(ratingValues), len(ratingValues)), dtype=np.float32)
for i in range(len(ratingValues) - 1):
    for j in range(i + 1, len(ratingValues)):
        userSimMatrix[i, j] = calCosineSimilarity(ratingValues[i], ratingValues[j])
        userSimMatrix[j, i] = userSimMatrix[i, j]


#接下来，我们要找到与每个用户最相近的K个用户，用这K个用户的喜好来对目标用户进行物品推荐，这里K=10
#这里我们选择最相近的10个用户
userMostSimDict = dict()
for i in range(len(ratingValues)):
    userMostSimDict[i] = sorted(enumerate(list(userSimMatrix[i])), key=lambda x: x[1], reverse=True)[:2]

# 用这K个用户的喜好中目标用户没有看过的电影进行推荐
userRecommendValues = np.zeros((len(ratingValues), len(ratingValues[0])), dtype=np.float32)  #

for i in range(len(ratingValues)):
    for j in range(len(ratingValues[i])):
        if ratingValues[i][j] == 0:
            val = 0
            for (user, sim) in userMostSimDict[i]:
                val += (ratingValues[user][j] * sim)
            userRecommendValues[i, j] = val

#为每个用户推荐10部电影：
userRecommendDict = dict()
for i in range(len(ratingValues)):
    userRecommendDict[i] = sorted(enumerate(list(userRecommendValues[i])), key=lambda x: x[1], reverse=True)[:2]



userRecommendList = []
for key, value in userRecommendDict.items():
    user = usersMap[key]
    for (movieId, val) in value:
        userRecommendList.append([user, moviesMap[movieId],val])


# 将推荐结果的电影id转换成对应的电影名
recommendDF = pd.DataFrame(userRecommendList, columns=['userId', 'movieId','Recommend_Val'])
recommendDF = pd.merge(recommendDF, moviesDF[['movieId', 'title']], on='movieId', how='inner')
recommendDF = recommendDF.sort_values(by=['userId'])


#print(recommendDF)

recommendDF.to_csv('userCF_Result.csv', index=False, header=True)


























from nltk import WordPunctTokenizer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import re
import pandas as pd
from sklearn import metrics
import implement_logistic


# read data
def create_data():
    data = pd.read_csv("train.csv", lineterminator="\n")
    train_data = data[:6000]
    test_data = data[6000:len(data)]
    print(train_data[:20])
    return train_data, test_data


# pre process train_data
def pre_process(data):
    clean_reviews = []

    # 需要移除的无用符号
    rmSignal = ['.', '#', '$', '?', '!', '%', ':/', ':', '-', '+', '/', '"']

    for comment_content in data['review']:
        texts = WordPunctTokenizer().tokenize(comment_content)   # NLTK分词

        text = [word.lower() for word in texts]
        for word in text:
            if word in rmSignal:                           # 去掉符号
                text.remove(word)
                continue

            if word.isdigit():                             # 移除数字
                text.remove(word)

            if re.search("[(.|…)]+", word):               # 移除分词不能识别的省略号
                text.remove(word)

        new_sentence = (" ".join(text))                     # change to string
        clean_reviews.append(new_sentence)

    return clean_reviews


def bag_of_words(train_reviews):
    vectorizer = CountVectorizer(max_features=10000)
    train_data_features = vectorizer.fit_transform(train_reviews)
    train_data_features = train_data_features.toarray()
    print(len(train_data_features[1]))
    return vectorizer, train_data_features


def train_model(train_data_features):
    logistic = LogisticRegression()
    logistic.fit(train_data_features, train_data["label"])

    return logistic


if __name__ == "__main__":
    train_data, test_data = create_data()

    clean_train_reviews = pre_process(train_data)           # 训练集文本预处理
    vector, train_features = bag_of_words(clean_train_reviews)          # 得到训练集的向量表示

    clean_test_reviews = pre_process(test_data)                    # 测试集文本预处理
    test_data_features = vector.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    system_model = train_model(train_features)  # sklearn自带的Logistic函数，训练模型

    system_result_proba_list = system_model.predict_proba(test_data_features)    # 预测结果
    emotion_list = []
    for result in system_result_proba_list:
        emotion_list.append(result[1])                              # 预测为正值的概率

    system_predict_roc_score = metrics.roc_auc_score(test_data["label"], emotion_list)
    print("The ROC score of the Logistic model in sklearn that comes with python is:", system_predict_roc_score)

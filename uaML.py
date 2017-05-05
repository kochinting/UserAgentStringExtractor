
"""
Data Engineer Coding Challenge
Chin-Ting Ko  04/22/2017

This program is to build a machine learning tool to extract/predict browser family and major version from user agents strings.
"""

from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

word_freq = {}
user_agent = []
ua_features = {}
ua_string = []
ua_family = []
ua_version = []
ua_final = []
test_ua_string = []
test_ua_family = []
test_ua_version = []

#"test_data_coding_exercise.txt"
def read_train_file():
    f = open("test_data_coding_exercise.txt", "r")
    line = f.readlines()[:1000]
#    line = f.readlines()
    for ua in line:
        ua_string.append(ua.split('\t')[0])
        ua_family.append(ua.split('\t')[1])
        ua_version.append(ua.split('\t')[2].rstrip())
    f.close()


def read_test_file():
    f = open("test_data_coding_exercise.txt", "r")
    line = f.readlines()
    for ua in line:
        test_ua_string.append(ua.split('\t')[0])
        test_ua_family.append(ua.split('\t')[1])
        test_ua_version.append(ua.split('\t')[2].rstrip())
    f.close()


def print_frequency():
    for word in ua_family:
        count = word_freq.get(word, 0)
        word_freq[word] = count+1
    frequency_list = word_freq.keys()
    for words in frequency_list:
        print words, word_freq[words]


def parse_ua():
    for features in ua_string:
        mozilla = features.split()[0]
        ua_features['Mozilla'] = mozilla
        #print "Mozilla: " + mozilla

        index1 = features.find('(')
        index2 = features.find(')')+1
        system = features[index1:index2]
        ua_features['System'] = system
        #print "System: "+system

        index3 = features.find('(', index2)
        index4 = features.find(')', index2)+1
        platfrom= features[index2+1: index3]
        ua_features['Platform'] = platfrom
        #print "Platform: "+platfrom
        platform_details = features[index3:index4]
        ua_features['Platform_Details'] = platform_details
        #print "Platform Details: "+platform_details

        extensions= features[features.rfind(")")+1:]
        ua_features['Extensions'] = extensions
        #print "Extensions: "
        #print extensions
        #print ua_features
        user_agent.append(ua_features.copy())


def ua_processed():
    for uastring, uafamily, uaversion, family_predicted, version_predicted in zip(ua_string, ua_family, ua_version, predict_family, predict_version):
        ua_final.append((uastring, uafamily, uaversion, family_predicted, version_predicted))


def write_file():
    with open("prediction_results.txt", "w") as result:
        for words in ua_final:
            result.write(str(words).strip('[]').strip('()').replace(', ', '\t').replace("'", "") + '\n')


def predict (X, Y):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X)
    tf_transformer_X = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer_X.transform(X_train_counts)
    tfidf_transformer_X = TfidfTransformer()
    X_train_tfidf = tfidf_transformer_X.fit_transform(X_train_counts)

    model = BernoulliNB()
    #model = MultinomialNB()
    #model = svm.SVC()
    model.fit(X_train_tfidf, Y)
    print "Classifier trained."
    print ""

    expected = Y
    predicted = model.predict(X_train_tfidf)
    print(metrics.classification_report(expected, predicted))
    #print(metrics.confusion_matrix(expected, predicted))
    return predicted.tolist()


read_train_file()
#read_test_file()
#print_frequency()
#print ""
parse_ua()

#print user_agent
#print ua_string
#print ua_family
#print ua_version


X = []
for lists in user_agent:
    X.append(lists['Extensions'])

#for lists in user_agent:
#    X.append(lists['Extensions']+lists['System'])

predict_family = list(predict(X, ua_family))
#print predict_family
predict_version = list(predict(X, ua_version))
#print predict_version


ua_processed()
write_file()

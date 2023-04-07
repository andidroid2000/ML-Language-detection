from sklearn import naive_bayes, svm, ensemble
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
import csv

# initialise the storage for the data that will be read from the files
trainSamples, trainLabels, validationSamples, validationLabels, testSamples, ids = [], [], [], [], [], []

# initialise the classifiers that will be used to predict the result for the given set of data (testSamples)
# and add them to a list for easier access
MNB = naive_bayes.MultinomialNB()
CNB = ComplementNB()
GNB = GaussianNB()
SGDC = SGDClassifier()
SVM = svm.SVC()
MLPC = MLPClassifier(verbose=True, early_stopping=True)
PERCEPTRON = SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty=None)
classifiers = [MNB, CNB, GNB, SGDC, SVM, MLPC, PERCEPTRON]

# store repetitive data in a dictionary for easier access
# for each file, store its path and list where raw data will be read
data = {
    'trainSamples': {'path': './ml-2022-unibuc-23-25/train_samples.txt', 'list': trainSamples},
    'trainLabels': {'path': './ml-2022-unibuc-23-25/train_labels.txt', 'list': trainLabels},
    'validationSamples': {'path': './ml-2022-unibuc-23-25/validation_samples.txt', 'list': validationSamples},
    'validationLabel': {'path': './ml-2022-unibuc-23-25/validation_labels.txt', 'list': validationLabels},
    'testSamples': {'path': './ml-2022-unibuc-23-25/test_samples.txt', 'list': testSamples}
}
# store other auxiliary paths used in the code
paths = {
    'testModelPath': './ml-2022-unibuc-23-25/ResultModel.txt',  # a file used to store results of model testing
    'testBestModelsPath': './ml-2022-unibuc-23-25/test.txt',  # a file used to store results of NB models testing
    'resultPath': './ml-2022-unibuc-23-25/RESULT.csv'  # stores the predictions
}


def getRawData(fileName):  # a function for reading data from each given file into a list
    fileDescriptor = open(data[fileName]['path'], encoding="utf-8")
    for line in fileDescriptor.readlines():
        # store the ids of the sentences from testSamples file to write them in the result file
        if fileName == 'testSamples':
            ids.append(line[0:6])
        # get rid of useless data from the file: ids, \n etc
        line = line[7:]
        # add the read data into its specific list
        data[fileName]['list'].append(line.strip("\n"))
    fileDescriptor.close()


def train(classifier, X, V):  # a function for training the model and checking its performance score
    classifier.fit(X, trainLabels)
    return classifier.score(V, validationLabels)


def writeResult(predictions):  # a function for writing the result data of the prediction into a csv
    n = open(paths['resultPath'], 'w', newline='')
    writer = csv.writer(n)
    header = ['id', 'label']
    writer.writerow(header)
    i = 0
    for prediction in predictions:
        # append the id of the sentence to the prediction made
        line = [ids[i], prediction]
        i += 1
        writer.writerow(line)


if __name__ == "__main__":
    # take the data from the given files
    for key in data.keys():
        getRawData(key)
    option = 3
    if option == 1:  # determinate the best classifiers for text
        fileDescriptor = open(paths['testModelPath'], 'a')
        for classifier in classifiers:
            # instatiate through a list of different types of classifiers to see which are the best ones
            vectorizer = CountVectorizer()
            # standardise the training data and get features of it
            X = vectorizer.fit_transform(trainSamples)
            # standardise the validation and test data
            V = vectorizer.transform(validationSamples)
            T = vectorizer.transform(testSamples)
            score = train(classifier, X, V)
            fileDescriptor.write(str(classifier) + ' : ' + str(score) + '\n')
    if option == 2:  # compute the best arguments for the best classifiers
        BestClassifiers = [MNB, CNB]
        #  MLPC is out as it takes too long to run and uses lots of resources
        fileDescriptor = open(paths['testBestModelsPath'], 'a')
        for classifier in BestClassifiers:
            # define the range for ngrams up to 6
            for i in range(1, 7):
                # define the range for trimming the most used words from the vocabulary
                # => found in10%, 20% .. up to 90% of sentences
                for j in range(1, 10):
                    p = j/10
                    vectorizer = CountVectorizer(ngram_range=(1, i), max_df=p, analyzer='char_wb', strip_accents='unicode')
                    # standardise the training data and get features of it
                    X = vectorizer.fit_transform(trainSamples)
                    # standardise the validation and test data
                    V = vectorizer.transform(validationSamples)
                    T = vectorizer.transform(testSamples)
                    # get the score for each possibility and write it to the text file
                    score = train(classifier, X, V)
                    fileDescriptor.write(str(classifier) + ' : ' + str(score) + ' with CountVectorizer(ngram_range=(1,' + str(i) + '), max_df=' + str(p) + '\n')
    if option == 3:
        vectorizer = CountVectorizer(ngram_range=(1, 5), max_df=0.2, analyzer='char_wb', strip_accents='unicode') # 0.725
        # for MNB ngram_range=(1, 5), max_df=0.1 and the others the same as CNB => 0.7162, best case for MNB
        # standardise the training data and get features of it
        X = vectorizer.fit_transform(trainSamples)
        # standardise the validation and test data
        V = vectorizer.transform(validationSamples)
        T = vectorizer.transform(testSamples)
        # compute the final score
        score = train(CNB, X, V)
        print(score)
        predictionsTest = CNB.predict(T)
        predictionsValidation = CNB.predict(V)
        # create the confusion matrix for best case train
        confusionM = confusion_matrix(validationLabels, predictionsValidation)
        print(confusionM)
        writeResult(predictionsTest)

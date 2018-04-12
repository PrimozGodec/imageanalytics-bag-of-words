import os
import random
import sys

import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

path = sys.argv[1]
dictionarySize = 50
random_seed = 42  # allow reproducibility

# load images and shuffle them
categories = os.listdir(path)
images_l = []

for cat in categories:
    images = os.listdir(os.path.join(path, cat))
    for im in images:
        images_l.append([os.path.join(path, cat, im),
                         categories.index(cat)])
random.Random(random_seed).shuffle(images_l)

# 10 fold-cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=random_seed)
predictions = []
trues = []

# for each fold
for train_index, test_index in kf.split(images_l):

    bow = cv2.BOWKMeansTrainer(dictionarySize)  # bag of words
    sift = cv2.xfeatures2d.SIFT_create()  # SIFT

    # build bow codebook
    for i in train_index:
        img = cv2.imread(images_l[i][0])
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        kp, des = sift.detectAndCompute(gray, None)
        if des is not None:
            bow.add(des)

    dictionary = bow.cluster()

    flann_params = dict(algorithm=1, trees=5)
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    bow_extract = cv2.BOWImgDescriptorExtractor(sift, matcher)
    bow_extract.setVocabulary(dictionary)  # the 64x20 dictionary, you made before

    # extract bow features for training set
    traindata = []
    trainlabels = []

    for i in train_index:
        img = cv2.imread(images_l[i][0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # get keypoints
        siftkp = sift.detect(gray)
        if len(siftkp) > 0:
            # let the bow extractor find descriptors,
            # and match them to the dictionary
            bowsig = bow_extract.compute(gray, siftkp)

            traindata.append(bowsig.tolist()[0])
            trainlabels.append(images_l[i][1])

    # extract bow features for test set
    testdata = []
    testlabels = []
    for id, i in enumerate(test_index):
        img = cv2.imread(images_l[i][0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # get keypoints
        siftkp = sift.detect(gray)
        if len(siftkp) > 0:
            # let the bow extractor find descriptors,
            # and match them to the dictionary
            bowsig = bow_extract.compute(gray, siftkp)
            testdata.append(bowsig.tolist()[0])
            testlabels.append(images_l[i][1])  # a number, from 0 to 20

    lr = LogisticRegression()
    lr.fit(traindata, trainlabels)
    pred = lr.predict(testdata)

    predictions += pred.tolist()
    trues += testlabels

print(f1_score(trues, predictions, average="weighted"))

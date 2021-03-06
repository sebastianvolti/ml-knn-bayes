import numpy as np
import pandas as pd
import sklearn
import utils

from attributes import CONTINUOUS_ATTRS, CATEGORICAL_ATTRS
from sklearn import naive_bayes, neighbors, preprocessing


def scikit_nb(training_examples, training_classes, nb_type='gaussian', m=1, **kwargs):
    if nb_type == 'gaussian':
        nb = naive_bayes.GaussianNB()
    elif nb_type == 'categorical':
        nb = naive_bayes.CategoricalNB(alpha=m)
    else:
        raise ValueError("nb_type value must be either 'gaussian' or 'categorical'")

    nb = nb.fit(training_examples, training_classes.values.ravel())

    return nb


def mixed_nb(training_examples, training_classes, m=1, **kwargs):
    cont_nb = scikit_nb(training_examples[CONTINUOUS_ATTRS], training_classes, nb_type='gaussian')
    cat_nb = scikit_nb(training_examples[CATEGORICAL_ATTRS], training_classes, nb_type='categorical', m=m)

    return cont_nb, cat_nb


def scikit_knn(training_examples, training_classes, k=5, **kwargs):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k).fit(training_examples, training_classes.values.ravel())

    return knn


def classify_examples(classifier, evaluation_examples):
    obtained_classes = pd.DataFrame(classifier.predict(evaluation_examples), columns=['classification'])

    return obtained_classes


def mixed_classify_examples(classifier_cont, classifier_cat, evaluation_examples):
    obtained_cont_classes = pd.DataFrame(classifier_cont.predict_proba(evaluation_examples[CONTINUOUS_ATTRS]), columns=['neg_classification', 'pos_classification'])
    obtained_cont_classes['classification'] = np.nan
    obtained_cat_classes = pd.DataFrame(classifier_cat.predict_proba(evaluation_examples[CATEGORICAL_ATTRS]), columns=['neg_classification', 'pos_classification'])
    obtained_cat_classes['classification'] = np.nan

    obtained_classes = obtained_cont_classes.multiply(obtained_cat_classes)

    obtained_classes.loc[obtained_classes.neg_classification >= obtained_classes.pos_classification, 'classification'] = 0
    obtained_classes.loc[obtained_classes.neg_classification < obtained_classes.pos_classification, 'classification'] = 1

    obtained_classes = obtained_classes['classification']

    return obtained_classes

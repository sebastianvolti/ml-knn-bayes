import copy
import numpy as np
import pandas as pd
import random
import utils
from sklearn.neighbors import KDTree


POS_LABEL = 1
NEG_LABEL = 0


def build_kdtree(training_examples):
    return KDTree(training_examples, metric='euclidean')


def knn(training_examples, training_classes, evaluation_examples, evaluation_classes, k):
    results_classes = copy.deepcopy(evaluation_classes)
    kdtree = build_kdtree(training_examples)
    for i in range(len(evaluation_examples)):
        example = evaluation_examples.loc[evaluation_examples.index == i]
        distances, indexes = kdtree.query(example, k)
        near_neighbors = []
        for elem in np.nditer(indexes):
            near_neighbors.append(elem)
        cont_minor_classes = cont_major_classes = 0
        for index in near_neighbors:
            classify_example = str(training_classes.loc[training_classes.index == int(index)]['classification'][index])
            if classify_example == str(NEG_LABEL):
                cont_minor_classes+=1
            elif classify_example == str(POS_LABEL):
                cont_major_classes+=1
        if (cont_minor_classes > cont_major_classes):
            example_result = NEG_LABEL
        else:
            example_result = POS_LABEL
        clasificator = evaluation_classes.loc[evaluation_classes.index == i]
        results_classes.at[i, 'classification']=example_result
    
    return results_classes, evaluation_classes

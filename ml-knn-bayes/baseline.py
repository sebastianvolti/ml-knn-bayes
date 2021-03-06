import copy
import numpy as np
import pandas as pd
import random
import utils

from sklearn.dummy import DummyClassifier


DATASET_FILE = './data/adult_dataset.csv'
DATASET_ATTRS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                 'hours-per-week', 'native-country']
DATASET_CLASS = 'classification'
DATASET_HEADERS = copy.deepcopy(DATASET_ATTRS) + ['classification']
SEED = 3
TS_PERCENTAGE = 0.8


random.seed(SEED)
np.random.seed(SEED) # Seteo la seed para numpy (que es lo que usa scikit)# Procesamiento de datos
dataset = pd.read_csv(DATASET_FILE, delimiter=', ', names=DATASET_HEADERS, header=None)
dataset = utils.replace_with_nan(dataset) # Antes del preprocesamiento. Es necesario para el conteo de instancias con valores faltantes
dataset = utils.preprocess_dataset(dataset)
training_set, evaluation_set = utils.partition_sets(dataset, TS_PERCENTAGE)

training_examples, training_classes = utils.get_classes(training_set)
evaluation_examples, evaluation_classes = utils.get_classes(evaluation_set)


def baseline(training_examples, training_classes, evaluation_examples, evaluation_classes):
    clf = DummyClassifier(strategy='stratified')
    clf = clf.fit(training_examples, training_classes)
    obtained_classes = clf.predict(evaluation_examples)

    acc = utils.get_accuracy(obtained_classes, evaluation_classes)
    prec = utils.get_precision(obtained_classes, evaluation_classes)
    rec = utils.get_recall(obtained_classes, evaluation_classes)
    f1 = utils.get_f1_score(obtained_classes, evaluation_classes)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'classification': obtained_classes # Ejemplos ya clasificados
    }

res = baseline(training_examples, training_classes, evaluation_examples, evaluation_classes)
print(res)

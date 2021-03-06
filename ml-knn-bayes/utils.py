import copy
import csv
import logging
import math
import numpy as np
import pandas as pd
import random

from attributes import CONTINUOUS_ATTRS, DATASET_ATTRS, DATASET_CLASS, DATASET_FILE, DATASET_HEADERS, DATASET_TEST
from sklearn import metrics, preprocessing


# Auxiliares para el preprocesamiento

def label(dataset, attrs):
    le = preprocessing.LabelEncoder()
    dataset[attrs] = dataset[attrs].apply(le.fit_transform)

    return dataset


def normalize(dataset):
    min_max_scaler = preprocessing.MinMaxScaler()
    dataset[DATASET_ATTRS] =  min_max_scaler.fit_transform(dataset[DATASET_ATTRS])
    return dataset

def discretize(dataset, attrs, bins):
    for attr in attrs:
        dataset[attr] = pd.qcut(dataset[attr], bins, duplicates='drop')


def replace_with_nan(dataset):
    return dataset.replace(to_replace='?', value=np.NaN)


def attributes_with_null(dataset):
    return [col_name for col_name, col in dataset.iteritems() if col.isna().sum() != 0]


def replace_most_common_val(dataset, attrs):
    for attr in attrs:
        most_common = dataset[attr].value_counts().index[0]
        dataset[attr] = dataset[attr].fillna(most_common)

def preprocess_dataset(dataset, discretization=0, normalizer=0):
    dataset = replace_with_nan(dataset)
    attrs = attributes_with_null(dataset)
    replace_most_common_val(dataset, attrs)
    
    if discretization > 0:
        discretize(dataset, CONTINUOUS_ATTRS, discretization)

    dataset = label(dataset, DATASET_ATTRS + [DATASET_CLASS])

    if normalizer > 0:
       dataset = normalize(dataset)

    return dataset.drop(DATASET_TEST, axis=1)


# Auxiliares para particionar el dataset

def partition_sets(data, training_set_percentage, shuffle=True):
    ''' Mezcla los datos de entrenamiento y los separo en conjunto de entrenamiento y validación '''
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True) # https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
    training_set_size = int(len(data) * training_set_percentage)
    training_set = data[:training_set_size].reset_index(drop=True)
    test_set = data[training_set_size:].reset_index(drop=True)
    return training_set, test_set


def vc_partition_folds(data, k):
    '''Separo conjunto de entrenamiento en k folds (k subconjuntos con misma cantidad de elementos)'''
    training_set_size = math.ceil(int((len(data)) * (1/k)))
    list_example_set = []
    for i in range(k):
        example_set = data[:training_set_size]
        list_example_set.insert(i, example_set)
        data = data[training_set_size:]
    return list_example_set


def get_classes(data):
    ''' Separa los datos de entrenamiento en dos listas distintas.
    Para cada ejemplo: [attr1, attr2, ..., attrN, class] en [attr1, attr2, ..., attrN] | [class] '''
    classes = data[['classification']].copy(deep=True)
    examples = data.drop(columns='classification')

    return examples, classes


# Auxiliares para métricas

def get_accuracy(obtained_classes, test_classes):
    return metrics.accuracy_score(test_classes, obtained_classes)


def get_precision_knn(obtained_classes, test_classes):
    return metrics.precision_score(test_classes, obtained_classes)


def get_recall_knn(obtained_classes, test_classes):
    return metrics.recall_score(test_classes, obtained_classes)


def get_f1_score_knn(obtained_classes, test_classes):
    return metrics.f1_score(test_classes, obtained_classes)


def get_precision(obtained_classes, test_classes):
    return metrics.precision_score(test_classes, obtained_classes)


def get_recall(obtained_classes, test_classes):
    return metrics.recall_score(test_classes, obtained_classes)


def get_f1_score(obtained_classes, test_classes):
    return metrics.f1_score(test_classes, obtained_classes)


def get_metrics(obtained_classes, evaluation_classes):
    acc = get_accuracy(obtained_classes, evaluation_classes)
    prec = get_precision(obtained_classes, evaluation_classes)
    rec = get_recall(obtained_classes, evaluation_classes)
    f1 = get_f1_score(obtained_classes, evaluation_classes)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
    }


def print_metrics(data):
    logging.info(f"> Accuracy: {data['accuracy']*100:.2f}%")
    logging.info(f"> Precision: {data['precision']*100:.2f}%")
    logging.info(f"> Recall: {data['recall']*100:.2f}%")
    logging.info(f"> F1: {data['f1']*100:.2f}%")


def cross_val(classifier, classify_function, training_set_folds, cross_validation, **kwargs):
    # Aplicacion de validación cruzada k iteraciones
    acc_prom = 0
    pre_prom = 0
    rec_prom = 0
    f1_prom  = 0
    for i in range(len(training_set_folds)):
        test_set = []
        training_set = []
        test_set = training_set_folds[i].reset_index(drop=True)
        is_first = True
        for j in range(len(training_set_folds)):
            if (j != i):
                #aqui hay que armar el conj de entrenamiento para cada iteracion
                if (is_first):
                    training_set = training_set_folds[j]
                    is_first = False
                else:
                    training_set = training_set.append(training_set_folds[j])

        training_examples, training_classes = get_classes(training_set.reset_index(drop=True))
        evaluation_examples_cv, evaluation_classes_cv = get_classes(test_set)

        clfs = classifier(training_examples, training_classes, **kwargs)
        if type(clfs) != tuple:
            clfs = (clfs,)

        obtained_classes_cv = classify_function(*clfs, evaluation_examples_cv)

        metrics = get_metrics(obtained_classes_cv, evaluation_classes_cv)
        acc_prom += metrics['accuracy']
        pre_prom += metrics['precision']
        rec_prom += metrics['recall']
        f1_prom  += metrics['f1']      

    metrics = {
        'accuracy': acc_prom/cross_validation,
        'precision': pre_prom/cross_validation,
        'recall': rec_prom/cross_validation,
        'f1': f1_prom/cross_validation
    }
    
    return metrics


def prepare_data(DISCR, NORM, TSET_PERCENTAGE):
    # Lectura del csv
    dataset = pd.read_csv(DATASET_FILE, delimiter=', ', names=DATASET_HEADERS, header=None, engine='python')

    # Preprocesamiento de datos 
    dataset = preprocess_dataset(dataset, discretization=DISCR, normalizer=NORM)

    # Particionamiento de conjuntos
    training_set, evaluation_set = partition_sets(dataset, TSET_PERCENTAGE)

    # Separo los conjuntos en atributos | clase
    training_examples, training_classes = get_classes(training_set)
    evaluation_examples, evaluation_classes = get_classes(evaluation_set)

    return (training_examples, training_classes, evaluation_examples, evaluation_classes, training_set, evaluation_set)
import copy
import numpy as np
import pandas as pd
import utils

from scipy import stats
from sklearn import preprocessing
from attributes import CONTINUOUS_ATTRS, CATEGORICAL_ATTRS, DATASET_ATTRS, DATASET_CLASS, DATASET_HEADERS


DATASET_FILE = './data/adult_dataset.csv'


def normal_test(dataset):
    ps = []

    for attr in DATASET_ATTRS:
        k, p = stats.normaltest(dataset[attr])
        ps.append((attr, p))

    return ps


def get_analysis(dataset, attr_names, class_name):
    possible_values = {}

    # Posibles valores de cada atributo
    for attr in attr_names:
        possible_values[attr] = dataset[attr].unique()

    # Cantidad de valores faltantes por atributo
    missing_values_per_attr = dataset.isna().sum()

    # Cantidad de valores faltantes
    missing_values = dataset.isna().sum().sum()

    # Cantidad de instancias con valores faltantes
    missing_instances = dataset.shape[0] - dataset.dropna().shape[0] # dropna() remueve las filas con valores NaN

    analysis = {
        'instances': dataset.shape[0],          # Cantidad de instancias totales del dataset
        'attributes': dataset.shape[1],     # Cantidad de atributos totales del dataset
        'class_distribution': dataset[class_name].value_counts(),   # Cantidad de clases >50 y <=50
        'possible_values': possible_values,     # Posibles valores que puede tomar cada atributo
        'missing_values_per_attr': missing_values_per_attr,
        'missing_values': missing_values,
        'missing_instances': missing_instances
    }
    
    return analysis

def main():
    dataset = pd.read_csv(DATASET_FILE, delimiter=', ', names=DATASET_HEADERS, header=None)
    dataset = utils.replace_with_nan(dataset) # Antes del preprocesamiento. Es necesario para el conteo de instancias con valores faltantes
    analysis = get_analysis(dataset, DATASET_ATTRS, DATASET_CLASS)
    print(analysis)

    # Testing estadístico para ver si las distribuciones son normales
    # Preproceso el dataset
    dataset = utils.preprocess_dataset(dataset)
    
    # Hago un test de normalidad con el dataset inicial
    print("")
    print("")
    print("Resultados de tests de normalidad con el dataset")
    print("------------------------------------------------")
    ps = normal_test(dataset)
    for attr, p in ps:
        print(attr, p)
    print("")


if __name__ == "__main__":
   main()

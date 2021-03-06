import argparse
import copy
import logging
import numpy as np
import pandas as pd
import pprint
import random
import utils

import knn
import naive_bayes
import scikit

from attributes import CONTINUOUS_ATTRS, CATEGORICAL_ATTRS, DATASET_ATTRS, DATASET_CLASS, DATASET_FILE, DATASET_HEADERS
from sklearn.model_selection import cross_val_score


################## Config básica ##################

DEBUG = True

# Lee argumentos
ap = argparse.ArgumentParser(
    description='Tarea de AprendAut')
ap.add_argument('-s', '--seed', help='Indica la semilla a utilizar para la librería random')
ap.add_argument('-t', '--tset_percentage', help='Indica con un valor entre 0 y 1, qué porcentaje de los datos se usan para el entrenamiento (el resto es para validación')
ap.add_argument('-c', '--cross_validation', default=0,help='Si c > No solo decrece la Accuracy sino que decrece la metrica F1, recordemos que esta le da igual importancia a la precision y al recall. Esta baja de sebe a que el valor del recall decrece. utiliza validación cruzada de "c" iteraciones')
ap.add_argument('-i', '--prep_discretization', default=0, help='Si i > 0, se preprocesan los valores numéricos del dataset, discretizándolos en i bins')
ap.add_argument('-n', '--normalizer', default=0, help='Si n > 0, se preprocesan los valores numéricos del dataset, normalizandolos entre 0 y 1')
ap.add_argument('-p', '--part', help='Indica con valores a, b o c, qué parte de la tarea se quiere ejecutar')
ap.add_argument('-d', '--debug_level', default=2, help='0 si no se quiere imprimir nada, 1 para mensajes de info, 2 para mensajes de debug')
ap.add_argument('-k', '--knn_neighbors', default=3, help='k indica la cantidad de vecinos mas cercanos que utiliza knn')
ap.add_argument('-m', '--m_nb', default=1, help='m indica el parámetro de suavizado categórico para Naive Bayes')


args = vars(ap.parse_args())
seed = int(args['seed'])
training_set_percentage = float(args['tset_percentage'])
cross_validation = int(args['cross_validation'])
discretization = int(args['prep_discretization'])
part = args['part']
debug_level = int(args['debug_level'])
knn_neighbors = int(args['knn_neighbors'])
m_nb = float(args['m_nb'])
normalizer = int(args['normalizer'])


if debug_level == 0:
    logging_level = logging.WARNING
elif debug_level == 1:
    logging_level = logging.INFO
elif debug_level == 2:
    logging_level = logging.DEBUG

logging.basicConfig(level=logging_level, format='%(message)s')


################## Comienzo del main ##################

def main():

    random.seed(seed)
    np.random.seed(seed) # Seteo la seed para numpy (que es lo que usa scikit)

    # Lectura del csv
    dataset = pd.read_csv(DATASET_FILE, delimiter=', ', names=DATASET_HEADERS, header=None, engine='python')

    # Preprocesamiento de datos 
    dataset = utils.preprocess_dataset(dataset, discretization=discretization, normalizer=normalizer)

    # Particionamiento de datos
    training_set, evaluation_set = utils.partition_sets(dataset, training_set_percentage)
    if (cross_validation > 0):
        training_set_folds = utils.vc_partition_folds(training_set, cross_validation)
    else:
        # Separo los conjuntos en atributos | clase
        training_examples, training_classes = utils.get_classes(training_set)
        evaluation_examples, evaluation_classes = utils.get_classes(evaluation_set)
    if part == 'c':
        if (cross_validation > 0):
            # Aplicacion de validacipn cruzada k iteraciones
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
                logging.info(f"Iteración {i + 1}")
                logging.info(f"Construyendo clasificador con KNN nuestro...")
                training_examples, training_classes = utils.get_classes(training_set.reset_index(drop=True))
                evaluation_examples_cv, evaluation_classes_cv = utils.get_classes(test_set)
                results_classes_cv ,evaluation_classes_cv = knn.knn(training_examples, training_classes, evaluation_examples_cv, evaluation_classes_cv, knn_neighbors)
                metrics = utils.get_metrics(results_classes_cv, evaluation_classes_cv)
                utils.print_metrics(metrics)
                acc_prom += metrics['accuracy']
                pre_prom += metrics['precision']
                rec_prom += metrics['recall']
                f1_prom  += metrics['f1']      
                logging.info(f"KNN nuestro construido") 
                logging.info("")

            acc_prom = acc_prom/cross_validation
            pre_prom = pre_prom/cross_validation
            rec_prom = rec_prom/cross_validation
            f1_prom  = f1_prom/cross_validation
            print("Resultados PROMEDIO cross validation knn nuestro:")   
            print(" =>")
            print(f"Accuracy: {acc_prom*100:.2f}%")
            print(f"Precision: {pre_prom*100:.2f}%")
            print(f"Recall: {rec_prom*100:.2f}%")
            print(f"F1: {f1_prom*100:.2f}%")
        else:    
            logging.info(f"Construyendo clasificador con KNN nuestro...")
            results_classes ,evaluation_classes = knn.knn(training_examples, training_classes, evaluation_examples, evaluation_classes, knn_neighbors)
            metrics = utils.get_metrics(results_classes, evaluation_classes)
            utils.print_metrics(metrics)
            logging.info(f"KNN nuestro construido") 


    elif part == 'd':
        DATASET_LENGTH = len(DATASET_ATTRS)
        correct=0
        incorrect=0
        false_positive=0
        false_negatives=0
        positive=0
        negative=0
        t,classes=utils.get_classes(evaluation_set)
        mean_std=training_set.groupby('classification').agg([np.mean, np.std])
        likelihood ={}
        test_classes=[]
        prior=training_set.groupby('classification').size()
        for i in range(DATASET_LENGTH):
            likelihood[i]=training_set.groupby(['classification',DATASET_ATTRS[i]]).size().div(prior)
        
        print("Los valores posibles para cada att real son:")
        for i in range(len(CONTINUOUS_ATTRS)):
            print(CONTINUOUS_ATTRS[i])
            print(len(dataset[CONTINUOUS_ATTRS[i]].unique()))



        for i in range(evaluation_set.shape[0]):
            instance=evaluation_set.iloc[i]
            classification=naive_bayes.nb(training_set,instance,likelihood,mean_std,False,m_nb)
            test_classes.append(classification)
            if instance[DATASET_LENGTH]==1:
                if instance[DATASET_LENGTH]==classification:
                    correct+= 1
                    positive+= 1
                else:
                    incorrect+=1
                    false_negatives+=1 #Ejemplos positivos que se clasificaron como negativos osea falsos negativos
            else:
                if instance[DATASET_LENGTH]==classification:
                    correct+= 1
                    negative+=1
                else:
                    incorrect+=1
                    false_positive+=1 #Ejemplos negativos que se clasificaron como positivos osea falsos positivos

        metrics = utils.get_metrics(test_classes, classes)
        utils.print_metrics(metrics)  

    elif part == 'f':
        #validacion cruzada para Bayes
        if (cross_validation > 0):
            # Aplicacion de validacipn cruzada k iteraciones
            acc_prom = 0
            DATASET_LENGTH = len(DATASET_ATTRS)
            pre_prom = 0
            rec_prom = 0
            f1_prom  = 0
            index_m = -1
            m=[0.4,5,10,100,1500]
            for i in range(len(training_set_folds)):
                test_set = []
                training_set = []
                test_set = training_set_folds[i].reset_index(drop=True)
                is_first = True
                index_m+=1
                for j in range(len(training_set_folds)):
                    if (j != i):
                        #aqui hay que armar el conj de entrenamiento para cada iteracion
                        if (is_first):
                            training_set = training_set_folds[j]
                            is_first = False
                        else:
                            training_set = training_set.append(training_set_folds[j])
                logging.info(f"Iteración {i + 1}")
                logging.info(f"Construyendo clasificador con bayes...")
                
                classes=[]
                t,classes=utils.get_classes(test_set)
                mean_std={}
                likelihood ={}
                test_classes=[]
                mean_std=training_set.groupby('classification').agg([np.mean, np.std])
                prior=training_set.groupby('classification').size()
                for i in range(DATASET_LENGTH):
                    likelihood[i]=training_set.groupby(['classification',DATASET_ATTRS[i]]).size().div(prior)         
                for t in range(test_set.shape[0]):
                    instance=test_set.iloc[t]
                    classification=naive_bayes.nb(training_set,instance,likelihood,mean_std,True,m[index_m])
                    test_classes.append(classification)
                
                metrics = utils.get_metrics(test_classes, classes)
                utils.print_metrics(metrics)
                acc_prom += metrics['accuracy']
                pre_prom += metrics['precision']
                rec_prom += metrics['recall']
                f1_prom  += metrics['f1']      
                logging.info(f"bayes nuestro construido") 
                logging.info("")

            acc_prom = acc_prom/cross_validation
            pre_prom = pre_prom/cross_validation
            rec_prom = rec_prom/cross_validation
            f1_prom  = f1_prom/cross_validation
            print("Resultados PROMEDIO cross validation bayes:")   
            print(" =>")
            print(f"Accuracy: {acc_prom*100:.2f}%")
            print(f"Precision: {pre_prom*100:.2f}%")
            print(f"Recall: {rec_prom*100:.2f}%")
            print(f"F1: {f1_prom*100:.2f}%")

    elif part == 'scikit':
        training_examples, training_classes = utils.get_classes(training_set)
        evaluation_examples, evaluation_classes = utils.get_classes(evaluation_set)

        if cross_validation > 0:
            scores = ['accuracy', 'precision', 'recall', 'f1']

            # Cross validation para el k de KNN
            logging.info(f"Cross Validation - KNN de Scikit - k")
            ks = [1, 3, 5, 7, 9, 11, 13, 15]
            training_set_folds = utils.vc_partition_folds(training_set, cross_validation)
            for k in ks:
                metrics = {}
                print(f"K = {k}")
                metrics = utils.cross_val(scikit.scikit_knn, scikit.classify_examples, training_set_folds, cross_validation, k=k)
                utils.print_metrics(metrics)
                print("")
            logging.info(f"--------------------------------------------------")

            # Cross validation para el m de NB categórico
            logging.info(f"Cross Validation - NB Categórico de Scikit - m")
            ms=[0.001, 0.01, 0.1, 0.4, 1, 5, 10]
            training_set_folds = utils.vc_partition_folds(training_set, cross_validation)
            for m in ms:
                metrics = {}
                print(f"m = {m}")
                metrics = utils.cross_val(scikit.scikit_nb, scikit.classify_examples, training_set_folds, cross_validation, nb_type='categorical', m=m)
                utils.print_metrics(metrics)
                print("")
            logging.info(f"--------------------------------------------------")

            # Cross validation para el m de NB híbrido
            logging.info(f"Cross Validation - NB Híbrido de Scikit - m")
            ms=[0.001, 0.01, 0.1, 0.4, 1, 5, 10]
            training_set_folds = utils.vc_partition_folds(training_set, cross_validation)
            for m in ms:
                metrics = {}
                print(f"m = {m}")
                metrics = utils.cross_val(scikit.mixed_nb, scikit.mixed_classify_examples, training_set_folds, cross_validation, m=m)
                utils.print_metrics(metrics)
                print("")
        else:
            # Comparación NB contra Scikit
            classifier_nb_cont = scikit.scikit_nb(training_examples, training_classes, nb_type='gaussian')
            obtained_classes_nb = scikit.classify_examples(classifier_nb_cont, evaluation_examples)
            logging.info(f"")
            logging.info(f"Resultados Naive Bayes Gaussiano con Scikit")
            metrics = utils.get_metrics(obtained_classes_nb, evaluation_classes)
            utils.print_metrics(metrics)
            logging.info(f"")

            classifier_nb_cat = scikit.scikit_nb(training_examples, training_classes, nb_type='categorical', m=m_nb)
            obtained_classes_nb = scikit.classify_examples(classifier_nb_cat, evaluation_examples)
            logging.info(f"")
            logging.info(f"Resultados Naive Bayes Categórico con Scikit")
            metrics = utils.get_metrics(obtained_classes_nb, evaluation_classes)
            utils.print_metrics(metrics)
            logging.info(f"")

            classifier_nb_cont_mixed, classifier_nb_cat_mixed = scikit.mixed_nb(training_examples, training_classes, m=m_nb)
            obtained_classes_mixed_nb = scikit.mixed_classify_examples(classifier_nb_cont_mixed, classifier_nb_cat_mixed, evaluation_examples)
            logging.info(f"Resultados Naive Bayes mezclado Scikit")
            metrics = utils.get_metrics(obtained_classes_mixed_nb, evaluation_classes)
            utils.print_metrics(metrics)
            logging.info(f"")

            # Comparación KNN contra Scikit
            classifier_knn = scikit.scikit_knn(training_examples, training_classes, k=knn_neighbors)
            obtained_classes_knn = scikit.classify_examples(classifier_knn, evaluation_examples)
            logging.info(f"Resultados KNN Scikit")
            metrics = utils.get_metrics(obtained_classes_knn, evaluation_classes)
            utils.print_metrics(metrics)
    

if __name__ == "__main__":
   main()

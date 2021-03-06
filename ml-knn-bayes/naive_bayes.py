import numpy as np
import pandas as pd
import utils
import copy
import dataset_analysis
import math

from attributes import CONTINUOUS_ATTRS, DATASET_ATTRS


POS_LABEL = 1
NEG_LABEL = 0
DATASET_LENGTH = len(DATASET_ATTRS)


#negative -> 'classification' <=50
#positive -> 'classification' >50
def nb(dataset,instance,likelihood,mean_std,gauss,m):
    p_neg_value,p_pos_value=dataset.groupby('classification').size().div(len(dataset))
    negative,positive = dataset.groupby('classification').size()
    for i in range(DATASET_LENGTH):
        if instance[i] in likelihood[i][NEG_LABEL]:
            if DATASET_ATTRS[i] in CONTINUOUS_ATTRS and gauss:
                mean=mean_std[DATASET_ATTRS[i],'mean'][0]
                stdev=mean_std[DATASET_ATTRS[i],'std'][0]
                x=instance[DATASET_ATTRS[i]]
                exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
                p_neg_value=(( 1 / (math.sqrt(2 * math.pi) * stdev) )* exponent)*p_neg_value
            else:
                p_neg_value=(likelihood[i][NEG_LABEL][instance[i]])*p_neg_value
        else:
            k=len(dataset[DATASET_ATTRS[i]].unique())
            p=1/k
            p_neg_value=(m*p/(negative+m))*p_neg_value
        if instance[i] in likelihood[i][POS_LABEL]:
            if DATASET_ATTRS[i] in CONTINUOUS_ATTRS and gauss:
                mean=mean_std[DATASET_ATTRS[i],'mean'][1]
                stdev=mean_std[DATASET_ATTRS[i],'std'][1]
                x=instance[DATASET_ATTRS[i]]
                exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
                p_pos_value=(( 1 / (math.sqrt(2 * math.pi) * stdev) )* exponent)*p_pos_value
            else:
                p_pos_value=(likelihood[i][POS_LABEL][instance[i]])*p_pos_value
        else:
            k=len(dataset[DATASET_ATTRS[i]].unique())
            p=1/k
            p_pos_value=(m*p/(positive+m))*p_pos_value
        
    if (p_neg_value)>(p_pos_value):
        return NEG_LABEL
    else:
        return POS_LABEL


def full_nb(training_set, evaluation_set, m):
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
    for i in range(evaluation_set.shape[0]):
        instance=evaluation_set.iloc[i]
        classification=nb(training_set,instance,likelihood,mean_std,False,m)
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

    return test_classes

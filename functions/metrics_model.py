from fairlearn.metrics import MetricFrame
from fairlearn.metrics import false_positive_rate
from fairlearn.metrics import false_negative_rate
from fairlearn.metrics import selection_rate
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score, accuracy_score
from pandas import DataFrame
import fairlearn
import aif360
import pandas as pd

def assess_my_results(y_test, y_pred, sensitive_features, model_name):
    gm_base = MetricFrame(metrics=accuracy_score,  y_true= y_test, y_pred= y_pred, sensitive_features = sensitive_features)
    FPR_base = MetricFrame(metrics=false_positive_rate,  y_true= y_test, y_pred= y_pred, sensitive_features = sensitive_features)
    FNR_base = MetricFrame(metrics=false_negative_rate,  y_true= y_test, y_pred= y_pred, sensitive_features = sensitive_features)
    F1_base = MetricFrame(metrics=f1_score,  y_true= y_test, y_pred= y_pred, sensitive_features = sensitive_features) 
    Precision_base = MetricFrame(metrics=precision_score,  y_true= y_test, y_pred= y_pred, sensitive_features = sensitive_features) 
    Recall_base = MetricFrame(metrics=recall_score,  y_true= y_test, y_pred= y_pred, sensitive_features = sensitive_features) 
    Selection_rate_base = MetricFrame(metrics=selection_rate, y_true= y_test, y_pred= y_pred, sensitive_features = sensitive_features)
    DP_difference_base = fairlearn.metrics.demographic_parity_difference(y_true = y_test, y_pred = y_pred, sensitive_features = sensitive_features, method='between_groups', sample_weight=None)
    EQODS_difference_base = fairlearn.metrics.equalized_odds_difference(y_true = y_test, y_pred = y_pred, sensitive_features = sensitive_features, method='between_groups', sample_weight=None)

    
    #Create DF with metrics 

    base_model_metrics = DataFrame(data = {'Accuracy score' : gm_base.overall,
                                    'False positive rate' :FPR_base.overall,
                                    'False negative rate' :FNR_base.overall,
                                    'F1 score' : F1_base.overall,
                                    'Precision score' : Precision_base.overall,
                                    'Recall score' : Recall_base.overall,
                                    'Selection rate': Selection_rate_base.overall,
                                    'Demographic parity difference' : DP_difference_base,
                                    'Equalized odds difference': EQODS_difference_base,
                                    'SR Female' : Selection_rate_base.by_group['Female'],
                                    'SR Male' : Selection_rate_base.by_group['Male'],
                                    'FPR Male' : FPR_base.by_group['Male'],
                                    'FPR Female' : FPR_base.by_group['Female'],
                                    'FNR Male' : FNR_base.by_group['Male'],
                                    'FNR Female' : FNR_base.by_group['Female']
                                    },
                                    index = [f'{model_name}'])
    base_model_metrics = base_model_metrics.T
    return base_model_metrics


def metrics_2(metrics_pr):

    min_pr = pd.DataFrame(metrics_pr.T.min(), columns=['min'])
    mean_pr = pd.DataFrame(metrics_pr.T.mean(), columns=['mean'])
    max_pr = pd.DataFrame(metrics_pr.T.max(), columns=['max'])
    metr_pr = pd.merge(min_pr,mean_pr, left_index=True, right_index=True)
    
    df= pd.merge(metr_pr ,max_pr, left_index=True, right_index=True) 
    return df


def make_df(metrics_base, name_model):
    box_model = metrics_base.T
    box_model.insert(15, "Model_name", [name_model, name_model, name_model], True)
    return box_model



def percentage_calc(part, whole):
  return 100 * float(part)/float(whole)
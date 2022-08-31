from functions.validation_tuner import tuner_df
import pandas as pd
from functions.validation_tuner import classifiers_df
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def intermodel_to_csv(N_SPLIT,model,df,Model_name,Dataset_number): 
    constraints = 'demographic_parity'
    N_SPLIT = N_SPLIT
    model = model
    df_tuner = tuner_df(df, model,N_SPLIT,constraints)
    box_df = df_tuner[0]
    metrics_base =  df_tuner[1]
    metrics_gd =  df_tuner[2]
    metrics_eg =  df_tuner[3]
    metrics_to =  df_tuner[4]
    model_comparison = pd.DataFrame(data = {f'Base_Model_{Model_name}' : metrics_base['mean'],       
                                            f'GridSearch_{Model_name}' :metrics_gd['mean'],
                                            f'ExponentiatedGradient_{Model_name}':metrics_eg['mean'],
                                            f'ThresholdOptimizer_{Model_name}' : metrics_to['mean'],
                                        
                                            })
    model_comparison = model_comparison .T
    model_comparison.to_csv(f'Result_csv/model_comparison_{Model_name}_{Dataset_number}.csv')
    box_df.to_csv(f'Result_csv/box_df_{Model_name}_{Dataset_number}.csv')
    print(f'\n exporting {Model_name} resutls to csv')
    #export to csv    


def intermodel_to_csv_two(N_SPLIT,df,Dataset_number): 
    N_SPLIT = N_SPLIT
    df_tuner = classifiers_df(df,N_SPLIT)
    box_df = df_tuner[0]
    metrics_al=  pd.DataFrame(df_tuner[1].T)
    metrics_mc =  pd.DataFrame(df_tuner[2].T)
    metrics_pr =  pd.DataFrame(df_tuner[3].T)
    
    model_comparison = pd.DataFrame(data = {    
                                            'AdversarialDebiasing' :metrics_al['mean'],
                                            'MetaFairClassifier':metrics_mc['mean'],
                                            'PrejudiceRemover' : metrics_pr['mean'],
                                        
                                            })
    model_comparison = model_comparison .T
    model_comparison.to_csv(f'Result_csv/model_comparison_{Dataset_number}.csv')
    box_df.to_csv(f'Result_csv/box_df_{Dataset_number}.csv')
    print(f'\n exporting resutls to csv')
    #export to csv  
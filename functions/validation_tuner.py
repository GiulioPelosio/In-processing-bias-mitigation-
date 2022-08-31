from functions.metrics_model import assess_my_results, metrics_2, make_df
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from fairlearn.reductions import GridSearch, ExponentiatedGradient, DemographicParity
from fairlearn.postprocessing import ThresholdOptimizer
import emoji 
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
import aif360
import tensorflow.compat.v1 as tf
from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.algorithms.inprocessing import MetaFairClassifier
from sklearn.preprocessing import StandardScaler
from keras.backend import clear_session
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)






def print_emoji(N_SPLIT):
    return (emoji.emojize(':folded_hands:') * N_SPLIT)
    











def tuner_df(df, model,N_SPLIT,constraints ):
    privileged_groups = [{'Sex': 1}]
    unprivileged_groups = [{'Sex': 0}]
    kf = KFold(n_splits=N_SPLIT, shuffle = True)
    print('Starting hyperparameter tunings')
    i=0
    metrics_base = pd.DataFrame()
    # iterate for the n of split 
    print(' - Base model', end=" ")
    for train_index, test_index in kf.split(df):
    # split the train and test dataset based on the index from the k fold
        train_cv = df.iloc[train_index]
        val_cv = df.iloc[test_index]
        sensitive_features_test_kv =  pd.DataFrame(val_cv,columns = df.columns[:-1])['Sex'].apply(lambda x: 'Female' if x <= 0 else 'Male')
        base_model = model.fit(train_cv.iloc[:,:-1], train_cv.iloc[:,-1])
        y_preds_base = base_model.predict(val_cv.iloc[:,:-1])
        base_model_df = assess_my_results(val_cv.iloc[:, -1], y_preds_base, sensitive_features_test_kv, 'Base_model')
        metrics_base[f'{i}'] = base_model_df
        i+=1
        metrics_base_box = metrics_2(metrics_base)
        metrics_base_box_one = make_df(metrics_base_box, "Base_model")
        print(emoji.emojize(':folded_hands:'), end = ' ')
        
        
        
    #TRESHOLDOPTIMIZER #########################################################################################################################################################################################
    metrics_to= pd.DataFrame()
    # iterate for the n of split 
    print('\n - Treshold optimizer', end=" ")
    for train_index, test_index in kf.split(df):
        # split the train and test dataset based on the index from the k fold
        train_cv = df.iloc[train_index]
        val_cv = df.iloc[test_index]
        sensitive_features_test_kv =  pd.DataFrame(val_cv,columns = df.columns[:-1])['Sex'].apply(lambda x: 'Female' if x <= 0 else 'Male').reset_index(drop = True)
        base_model = model.fit(train_cv.iloc[:,:-1].reset_index(drop = True), train_cv.iloc[:,-1].reset_index(drop = True))
        unmitigated_base =  base_model
        postprocess_est = ThresholdOptimizer(
                    estimator=unmitigated_base,
                    constraints= constraints,
                    objective="balanced_accuracy_score",
                    prefit=True,
                    predict_method='predict_proba')
        sensitive_features_train_kv   =  pd.DataFrame(train_cv,columns = df.columns[:-1])['Sex'].apply(lambda x: 'Female' if x <= 0 else 'Male').reset_index(drop = True) 
        postprocess_est.fit(train_cv.iloc[:,:-1].reset_index(drop = True), train_cv.iloc[:,-1].reset_index(drop = True), sensitive_features= sensitive_features_train_kv.reset_index(drop = True))
        y_preds_TO = postprocess_est.predict(val_cv.iloc[:,:-1].reset_index(drop = True), sensitive_features= sensitive_features_test_kv)
        Treshold_Optimizer_df = assess_my_results(val_cv.iloc[:, -1].reset_index(drop = True), y_preds_TO, sensitive_features_test_kv, 'Treshold_Optimizer')
        metrics_to[f'{i}'] = Treshold_Optimizer_df
        i+=1
        metrics_to_box = metrics_2(metrics_to)
        metrics_to_box_one = make_df(metrics_to_box, "Treshold_Optimizer")
        print(emoji.emojize(':folded_hands:'), end = ' ')
    
    #EXPONETIATED GRADIENT  #########################################################################################################################################################################################
    pd.options.mode.chained_assignment = None
    np.random.seed(0)  # set seed for consistent results with ExponentiatedGradient
    constraint = DemographicParity()
    classifier = model
    mitigator = ExponentiatedGradient(classifier, constraint)
    metrics_eg = pd.DataFrame()
    # iterate for the n of split
    print('\n - Exponentiated gradient', end=" ") 
    for train_index, test_index in kf.split(df):
        # split the train and test dataset based on the index from the k fold
        pd.options.mode.chained_assignment = None
        np.random.seed(0)  # set seed for consistent results with ExponentiatedGradient
        constraint = DemographicParity()
        classifier = model
        mitigator = ExponentiatedGradient(classifier, constraint)
        train_cv = df.iloc[train_index]
        val_cv = df.iloc[test_index]
        sensitive_features_test_kv =  pd.DataFrame(val_cv,columns = df.columns[:-1])['Sex'].apply(lambda x: 'Female' if x <= 0 else 'Male').reset_index(drop = True)
        sensitive_features_train_kv   =  pd.DataFrame(train_cv,columns = df.columns[:-1])['Sex'].apply(lambda x: 'Female' if x <= 0 else 'Male').reset_index(drop = True) 
        mitigator.fit(train_cv.iloc[:,:-1].reset_index(drop = True), train_cv.iloc[:,-1].reset_index(drop = True), sensitive_features= sensitive_features_train_kv)
        y_pred_EG = mitigator.predict(val_cv.iloc[:,:-1].reset_index(drop = True))
        ExponentiatedGradient_df = assess_my_results(val_cv.iloc[:, -1].reset_index(drop = True), y_pred_EG, sensitive_features_test_kv, 'ExponentiatedGradient')
        metrics_eg[f'{i}'] = ExponentiatedGradient_df
        i+=1 
        metrics_eg_box = metrics_2(metrics_eg)
        metrics_eg_box_one = make_df(metrics_eg_box, "ExponentiatedGradient")
        print(emoji.emojize(':folded_hands:'), end = ' ')
    metrics_gd = pd.DataFrame()
    # iterate for the n of split 
    print('\n - Grid search', end=" ") 
    for train_index, test_index in kf.split(df):
        
        # split the train and test dataset based on the index from the k fold
        train_cv = df.iloc[train_index]
        val_cv = df.iloc[test_index]
        sensitive_features_test_kv =  pd.DataFrame(val_cv,columns = df.columns[:-1])['Sex'].apply(lambda x: 'Female' if x <= 0 else 'Male')
        sensitive_features_train_kv  =  pd.DataFrame(train_cv,columns = df.columns[:-1])['Sex'].apply(lambda x: 'Female' if x <= 0 else 'Male').reset_index(drop = True) 
        
        sweep = GridSearch(
            model,
            constraints=DemographicParity(),
            grid_size=100,
        )
        sweep.fit(train_cv.iloc[:,:-1].reset_index(drop = True), train_cv.iloc[:,-1].reset_index(drop = True), sensitive_features= sensitive_features_train_kv)
        y_pred_gd = sweep.predict(val_cv.iloc[:,:-1].reset_index(drop = True))
        GridSearch_df = assess_my_results(val_cv.iloc[:, -1].reset_index(drop = True),   y_pred_gd, sensitive_features_test_kv, 'GridSearch')
        metrics_gd[f'{i}'] =   GridSearch_df
        i+=1 
        metrics_gd_box = metrics_2(metrics_gd)
        metrics_gd_box_one = make_df(metrics_gd_box, "GridSearch")
        print(emoji.emojize(':folded_hands:'), end = ' ')

    box_df = pd.concat([metrics_to_box_one, metrics_base_box_one,metrics_eg_box_one, metrics_gd_box_one])
    return [box_df,metrics_base_box, metrics_gd_box,  metrics_eg_box, metrics_to_box]




def classifiers_df(df,N_SPLIT):
    print('Starting fair classifier generation')
    privileged_groups = [{'Sex': 1}]
    unprivileged_groups = [{'Sex': 0}]

#AdversarialDebiasing####################################################################################################################################################################
        
    kf = KFold(n_splits=N_SPLIT, shuffle = True)
    i=0
    metrics_al = pd.DataFrame()
    tf.disable_eager_execution()
        # iterate for the n of split 
    for train_index, test_index in kf.split(df):
    
        tf.disable_eager_execution()
        clear_session()
        # split the train and test dataset based on the index from the k fold
        train_cv = df.iloc[train_index]
        val_cv = df.iloc[test_index]
        sess = tf.Session()
        binaryLabelDataset_train = aif360.datasets.BinaryLabelDataset(
            favorable_label= 1,
            unfavorable_label= 0,
            df= train_cv,
            label_names=['target'],
            protected_attribute_names=['Sex'])
        binaryLabelDataset_test = aif360.datasets.BinaryLabelDataset(
            favorable_label= 1,
            unfavorable_label= 0,
            df= val_cv, 
            label_names=['target'],
            protected_attribute_names=['Sex'])
        
        sensitive_features_test_cv =  pd.DataFrame(val_cv,columns = df.columns[:-1])['Sex'].apply(lambda x: 'Female' if x <= 0 else 'Male')
        debiased_model = AdversarialDebiasing(privileged_groups = privileged_groups,
                            unprivileged_groups = unprivileged_groups,
                            scope_name='debiased_classifier',
                            debias=True,                      
                            sess=sess, num_epochs=100,)
        
        debiased_model.fit(binaryLabelDataset_train)
        df_tf = debiased_model.predict(binaryLabelDataset_test)
        df_tf = df_tf.convert_to_dataframe()[0]
        y_pred_tf = df_tf['target']
        AdversarialDebiasing_df = assess_my_results(val_cv.iloc[:,-1], y_pred_tf, sensitive_features_test_cv, 'AdversarialDebiasing')
        metrics_al[f'{i}'] =   AdversarialDebiasing_df
        
        i+=1
        metrics_al_box = metrics_2(metrics_al)
        metrics_al_box = make_df(metrics_al_box, "AdversarialDebiasing")

    print('\n - AdversarialDebiasing', print_emoji(N_SPLIT))
        

    #MetaFairClassifier####################################################################################################################################################################
    kf = KFold(n_splits=N_SPLIT, shuffle = True)
    i=0

    metrics_mc = pd.DataFrame()
        # iterate for the n of split 
    print('\n - MetaFair Classifier', end=" ")
    for train_index, test_index in kf.split(df):
        # split the train and test dataset based on the index from the k fold
        train_cv = df.iloc[train_index]
        val_cv = df.iloc[test_index]

        binaryLabelDataset_train = aif360.datasets.BinaryLabelDataset(
            favorable_label= 1,
            unfavorable_label= 0,
            df= train_cv,
            label_names=['target'],
            protected_attribute_names=['Sex'])
        binaryLabelDataset_test = aif360.datasets.BinaryLabelDataset(
            favorable_label= 1,
            unfavorable_label= 0,
            df= val_cv, 
            label_names=['target'],
            protected_attribute_names=['Sex'])
        
        sensitive_features_test_cv =  pd.DataFrame(val_cv,columns = df.columns[:-1])['Sex'].apply(lambda x: 'Female' if x <= 0 else 'Male')
        debiased_model = MetaFairClassifier(tau=0.7, sensitive_attr="Sex", type="sr", seed = 14).fit(binaryLabelDataset_train)
        
        df_mf = debiased_model.predict(binaryLabelDataset_test)
        df_mf = df_mf.convert_to_dataframe()[0]
        y_pred_mf = df_mf['target']
        MetaFairClassifier_df = assess_my_results(val_cv.iloc[:,-1], y_pred_mf, sensitive_features_test_cv, 'MetaFairClassifier')
        metrics_mc[f'{i}'] =   MetaFairClassifier_df
        
        i+=1
        metrics_mc_box = metrics_2(metrics_mc)
        metrics_mc_box = make_df(metrics_mc_box, "MetaFairClassifier")
        metrics_mc_box 
        print(emoji.emojize(':folded_hands:'), end = ' ')



    kf = KFold(n_splits=N_SPLIT, shuffle = True)
    i=0
    metrics_pr = pd.DataFrame()
        # iterate for the n of split 
    print('\n - Prejudice Remover', end=" ")
    for train_index, test_index in kf.split(df):
        # split the train and test dataset based on the index from the k fold
        train_cv = df.iloc[train_index]
        val_cv = df.iloc[test_index]

        binaryLabelDataset_train = aif360.datasets.BinaryLabelDataset(
            favorable_label= 1,
            unfavorable_label= 0,
            df= train_cv,
            label_names=['target'],
            protected_attribute_names=['Sex'])
        binaryLabelDataset_test = aif360.datasets.BinaryLabelDataset(
            favorable_label= 1,
            unfavorable_label= 0,
            df= val_cv, 
            label_names=['target'],
            protected_attribute_names=['Sex'])
        
        sensitive_features_test =  pd.DataFrame(val_cv,columns = df.columns[:-1])['Sex'].apply(lambda x: 'Female' if x <= 0 else 'Male')
        model_1 = PrejudiceRemover(sensitive_attr="Sex", eta=25.0)
        pr_orig_scaler = StandardScaler()


        pr_orig_panel19 = model_1.fit(binaryLabelDataset_train)

        y_pred_pr = pr_orig_panel19.predict(binaryLabelDataset_test)



        y_pred_pr = y_pred_pr.convert_to_dataframe()[0]
        y_pred_pr= y_pred_pr['target']

        PrejudiceRemover_df = assess_my_results(val_cv.iloc[:,-1], y_pred_pr, sensitive_features_test, 'PrejudiceRemover')
        metrics_pr[f'{i}'] =     PrejudiceRemover_df
        
        i+=1
        metrics_pr_box = metrics_2(metrics_pr)
        metrics_pr_box = make_df(metrics_pr_box, "PrejudiceRemover")
        print(emoji.emojize(':folded_hands:'), end = ' ')
    
    box_df = pd.concat([metrics_al_box, metrics_mc_box, metrics_pr_box])
    return [box_df,metrics_al_box, metrics_mc_box,  metrics_pr_box]
















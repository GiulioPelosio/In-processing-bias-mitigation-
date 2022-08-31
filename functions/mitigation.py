from functions.csv_exporter import intermodel_to_csv
from functions.csv_exporter import intermodel_to_csv_two
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def mitigate(df,Dataset_number,N_SPLIT):
    Model_name_one = "LogisticRegression"
    print(f'{Model_name_one}model starting')
    model_one = LogisticRegression(solver='liblinear', fit_intercept=True)
    intermodel_to_csv(N_SPLIT,model_one,df,Model_name_one,Dataset_number)
    print(f'\n {Model_name_one} model completed')

    Model_name_two = "DecisionTree"
    print(f'\n {Model_name_two} model starting')
    model_two =  DecisionTreeClassifier(random_state=0)
    intermodel_to_csv(N_SPLIT,model_two,df,Model_name_two,Dataset_number)
    print(f'\n {Model_name_two} model completed')

    Model_name_three = "Random Forest"
    print(f'\n {Model_name_three} model starting')
    model_three = RandomForestClassifier(max_depth=2, random_state=0)
    intermodel_to_csv(N_SPLIT,model_three,df,Model_name_three,Dataset_number)
    print(f'\n {Model_name_three} model completed')
    
    Model_name_four = "XGB"
    print(f'\n {Model_name_four} model starting')
    model_four = XGBClassifier()
    intermodel_to_csv(N_SPLIT,model_four,df,Model_name_four,Dataset_number)
    print(f'\n {Model_name_four} model completed')
    
    print('\n generating classifier')
    intermodel_to_csv_two(N_SPLIT,df,Dataset_number)
    print('\n mitigation completed all results are stored in local directory')
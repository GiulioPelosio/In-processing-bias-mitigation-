
import pandas as pd
from sklearn.preprocessing import StandardScaler

def transform_data_ACS(data):
#convert income and sex to 0 and 1 
    data['PINCP'] = data['PINCP'].apply(lambda x: 0 if x <50000 else 1)
    data['SEX'] = data['SEX'].apply(lambda x: 0 if x == 2 else 1)

# change name of variables for Sex and income 
    data.rename(columns = {'PINCP':'target', 'SEX':'Sex',}, inplace = True)

    return data


def transform_data_adult(X_raw, Y):
    
    Income_column = pd.DataFrame(Y, columns = ['target'])
    Income_column .iloc[:,0] = Income_column .iloc[:,0].astype(int)
    df_adult = pd.concat([X_raw, Income_column ], axis =1)
    return df_adult

def transform_data_bank(data_one):
#convert income and sex to 0 and 1 

# change name of variables for Sex and income 
    data_one.rename(columns = {'Gender':'Sex', 'Approved':'target',}, inplace = True)

    return data_one



def transform_databank_churns(data_two):
#convert income and sex to 0 and 1 
    data_two['target'] = data_two['Income_Category'].apply(lambda x: 0 if x == 'Less than $40K' or x == '$40K - $60K' else 1)
    data_two['Gender'] = data_two['Gender'].apply(lambda x: 0 if x == 'F' else 1)

# change name of variables for Sex and income 
    data_two.rename(columns = {'Gender':'Sex'}, inplace = True)
#drop income column
    data_two.drop(['Income_Category'], axis=1)
    data_two.drop(['CLIENTNUM'], axis=1)

    return data_two


def scale_df(df):
    Income_column = pd.DataFrame(df, columns = ['target'])
    x = df.drop(['target'], axis = 1)
    scaler = StandardScaler()
    x_data = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
    df_scaled = pd.concat([x_data, Income_column ], axis =1)
    return df_scaled
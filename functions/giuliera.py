
from functions.graph_generator import box_graph_inter,  selection_comparison, data_sample_chart, number_of_individuals, disparity_sample_percetange, disparity_selection_percentage, data_selectionrate_chart
import pandas as pd
import plotly.express as px
import datapane as dp





code_one = """
$ git clone https://github.com/GiulioPelosio/GIULIERAFAIR.git
"""
code_two = """
import pandas as pd
from functions.giuliera import Giuliera_fair
from functions.mitigation import mitigate
"""
code_three = """
df = pd.read_csv()
"""
code_four = """
N_SPLIT = 2 
Dataset_number = 'ds_one'
mitigate(df,Dataset_number,N_SPLIT)
"""
code_five = """
mitigate(df,Dataset_number,N_SPLIT)
Giuliera_fair(df)
"""










def Giuliera_fair(df):
#PAGE 0NE ############################################
#DATA PAGE VARIABLES
    number_individuals = number_of_individuals(df)
    disparity_sample = disparity_sample_percetange(df)
    disparity_lable = disparity_selection_percentage(df)
#DATA CHARTS
    positive_lable_chart = data_selectionrate_chart(df)
    sample_chart = data_sample_chart(df)
#PAGE TWO ############################################
#MODEL NAME VARIABLES:
    Dataset_number = "DS_one"
    Model_name_one = "LogisticRegression"
    Model_name_two = "Random Forest"
    Model_name_three = "XGB"
    Model_name_four = "DecisionTree"

#IMPORT DATA FOR GRAPHS 
####MODEL 1
    comparison_df_logistic = pd.read_csv (f'Result_csv\model_comparison_{Model_name_one}_{Dataset_number}.csv' , index_col= 0)
    box_df_logistic = pd.read_csv (f'Result_csv/box_df_{Model_name_one}_{Dataset_number}.csv' , index_col= 0)
####MODEL 2
    comparison_df_randomforest = pd.read_csv (f'Result_csv\model_comparison_{Model_name_two}_{Dataset_number}.csv' , index_col= 0)
    box_df_randomforest = pd.read_csv (f'Result_csv/box_df_{Model_name_two}_{Dataset_number}.csv' , index_col= 0)
####MODEL 3
    comparison_df_xgb = pd.read_csv (f'Result_csv\model_comparison_{Model_name_three}_{Dataset_number}.csv' , index_col= 0)
    box_df_xgb = pd.read_csv (f'Result_csv/box_df_{Model_name_three}_{Dataset_number}.csv' , index_col= 0)
####MODEL 4
    comparison_df_DecisionTree = pd.read_csv (f'Result_csv\model_comparison_{Model_name_four}_{Dataset_number}.csv' , index_col= 0)
    box_df_DecisionTree = pd.read_csv (f'Result_csv/box_df_{Model_name_four}_{Dataset_number}.csv' , index_col= 0)
####CLASSIFIER
    box_df_classifier= pd.read_csv (f'Result_csv/box_df_{Dataset_number}.csv' , index_col= 0)
    comparison_df_classifier = pd.read_csv (f'Result_csv\model_comparison_{Dataset_number}.csv' , index_col= 0)
#BOX PLOTS#######################################################################################################
    ####MODEL 1 BOX
    box_chart_logistic = box_graph_inter(box_df_logistic)
    ####MODEL 2 BOX
    box_chart_randomforest = box_graph_inter(box_df_randomforest)
    ####MODEL 3 BOX
    box_chart_xgb = box_graph_inter(box_df_xgb)
    ####MODEL 4 BOX
    box_chart_DecisionTree = box_graph_inter(box_df_DecisionTree)
    ####MODEL 5 BOX
    box_chart_classifier = box_graph_inter(box_df_classifier)
#PARETO PLOTS#######################################################################################################
#TUNERS
#DATA 
    tuner_comparison = pd.concat([comparison_df_logistic,comparison_df_randomforest,comparison_df_xgb,comparison_df_DecisionTree])
    classifier_comparison = comparison_df_classifier
    overall_comparison = pd.concat([comparison_df_logistic,comparison_df_randomforest,comparison_df_xgb,comparison_df_DecisionTree,comparison_df_classifier])
#CHARTS
    tuner_pareto = px.scatter(tuner_comparison ,x= 'Accuracy score', y= 'Demographic parity difference', color = tuner_comparison .index, symbol = tuner_comparison.index,  size='Accuracy score',hover_data=['Accuracy score'])
    classifier_pareto = px.scatter(classifier_comparison ,x= 'Accuracy score', y= 'Demographic parity difference', color = classifier_comparison .index, symbol = classifier_comparison.index,  size='Accuracy score',hover_data=['Accuracy score'])
    overall_pareto = px.scatter(overall_comparison ,x= 'Accuracy score', y= 'Demographic parity difference', color = overall_comparison .index, symbol = overall_comparison.index,  size='Accuracy score',hover_data=['Accuracy score'])
#MODEL SELECTION CHARTS#####################################################################################################
    selection_logistic = selection_comparison(comparison_df_logistic)
    selection_randomforest = selection_comparison(comparison_df_randomforest)
    selection_xgb = selection_comparison(comparison_df_xgb)
    selection_DecisionTree = selection_comparison(comparison_df_DecisionTree)
    selection_classifier = selection_comparison(comparison_df_classifier)

#METRICS 

#comparison_df_classifier = comparison_df_classifier.drop(columns=['Model_name'])
    total_comparison = overall_comparison.drop('Model_name', axis=1)
    total_comparison.to_csv(f'Result_csv/total_comparison_{Dataset_number}.csv')
# Create scatter plot for model comparison 

#VARIABLE CODE FOR REPORT

#HTML REPORT################################################################################################################
    dp.Report(
        dp.Page(
            title="Home",
            blocks=[dp.Media(file="media\giulieralogo.bmp"),dp.Media(file="media\linea.bmp"), dp.Media(file="media\image.dim.180.natwest.png"),dp.Media(file="media\linea.bmp")]
        
        ),

        dp.Page(
            title="Data",
            blocks=[dp.Media(file="media\giulieralogo - Copy.bmp"), dp.Group(dp.BigNumber( heading="Number of individuals", value= number_individuals  ), dp.BigNumber( heading= "Disparity in sample", value= f'{ disparity_sample}%'), dp.BigNumber( heading="Disparity in positive leable", value= f'{disparity_lable}%'), columns=3),
            dp.Select(blocks={ dp.Plot(sample_chart, label='Sample by segment'), dp.Plot(positive_lable_chart, label='Positive lable by segment')}),
            dp.Media(file="media\linea.bmp")
            ]

        ),

        dp.Page(
            title="Model diagnostics",
            blocks=[dp.Media(file="media\giulieralogo - Copy.bmp"),
            dp.Media(file="media\linea - Copy.bmp"),
            dp.Select(blocks={ dp.Plot(box_chart_logistic, label='Logistic Regression'), dp.Plot(box_chart_DecisionTree, label='DecisionTree'),  dp.Plot(box_chart_randomforest, label='Random forest'), dp.Plot(box_chart_xgb, label='XGB'),dp.Plot(box_chart_classifier, label='IBM Classifiers')},type=dp.SelectType.DROPDOWN),
            dp.Media(file="media\linea - Copy.bmp"),
            dp.Select(blocks={ dp.Plot(tuner_pareto, label='Tuner comparison'), dp.Plot(classifier_pareto, label='Classifier comparison'), dp.Plot(overall_pareto, label='Overall comparison')}),
            dp.Media(file="media\linea - Copy.bmp"),
            dp.Select(blocks={ dp.Plot(selection_logistic, label='Logistic Regression'), dp.Plot(selection_DecisionTree, label='DecisionTree'),  dp.Plot(selection_randomforest, label='Random forest'), dp.Plot(selection_xgb  , label='XGB'),dp.Plot(selection_classifier, label='IBM Classifiers')},type=dp.SelectType.DROPDOWN),
            dp.Media(file="media\linea.bmp")
            ]
        
        ),
    

    dp.Page(
        title="Report",
        blocks=[total_comparison, dp.Attachment(file="Result_csv/total_comparison_DS_one.csv")]         

        ),


        
    dp.Page(
        title="Documentation",
        blocks=[dp.Media(file="media\giulieralogo - Copy.bmp"),dp.Media(file="media\linea - Copy.bmp"), dp.Text("### Clone the github repository into your local directory",),dp.Code(code=code_one, language="python"),
        dp.Text("### Import the following libraries and functions"), dp.Code(code=code_two, language="python"),
        dp.Text("### Import your dataset under the variable df"), dp.Code(code=code_three, language="python"),
        dp.Text("### Select number of split for K-fold validation and number of dataset for future reference"), dp.Code(code=code_four, language="python"),
        dp.Text("### Initiate mitigation and generate dashboard"), dp.Code(code=code_five, language="python")]

        ),         

    ).save(path='GIULIERAFAIR.html', open=True,formatting = dp.ReportFormatting(accent_color="purple"))
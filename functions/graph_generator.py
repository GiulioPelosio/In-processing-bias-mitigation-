

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from functions.metrics_model import percentage_calc


def box_graph_inter(df):
    
    fig = make_subplots(rows=1, cols=2)
    
    fig.add_trace(
        go.Box(x=df["Model_name"], y=df["Demographic parity difference"],name = 'Demographic parity',marker=dict( color='purple')),
        row=1, col=1, 
    )

    fig.add_trace(
        go.Box(x=df["Model_name"], y=df["Accuracy score"],name = 'Accuracy score',marker=dict( color='pink')),
        row=1, col=2, 

    )

    fig.update_layout(height=600, width=2000)
    return fig




def selection_comparison(comparison_df):
    months = comparison_df.index

    selection_bar = go.Figure()
    selection_bar .add_trace(go.Bar(
        x=months,
        y=comparison_df['SR Male'],
        name='Male',
        marker_color='purple'
    ))
    selection_bar .add_trace(go.Bar(
        x=months,
        y=comparison_df['SR Female'],
        name='Female',
        marker_color='pink'
    ))

    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    selection_bar.update_layout(barmode='group', xaxis_tickangle=-0)
    return selection_bar  



#EDA###################################################################################################################################


def data_sample_chart(df):
    male_sample = df[df['Sex'] == 1]
    male_sample_count = male_sample['Sex'].value_counts()

    female_sample = df[df['Sex'] == 0]
    female_sample_count = female_sample['Sex'].value_counts()

    data = pd.DataFrame({'Gender':['Female','Male'], 'Count' :[female_sample_count[0], male_sample_count[1]]})
    ds_sample_chart = px.bar(data, x='Count', y='Gender', color="Gender", color_discrete_map={"Male": 'purple',"Female": 'pink' })
    return ds_sample_chart

def data_selectionrate_chart(df):

    male_selection = df[(df['Sex'] ==1)&(df['target'] ==1)]
    male_selection = len(male_selection.index)

    female_selection = df[(df['Sex'] ==0)&(df['target'] ==1)]
    female_selection = len(female_selection.index)

    data = pd.DataFrame({'Gender':['Female','Male'], 'Count' :[female_selection, male_selection]})
    ds_selection_chart = px.bar(data, x='Count', y='Gender', color="Gender", color_discrete_map={"Male": 'purple',"Female": 'pink' })

    return ds_selection_chart 



def number_of_individuals(df):

    index = df.index
    number_of_rows = len(index)
    return number_of_rows


def disparity_sample_percetange(df):
    male_sample = df[df['Sex'] == 1]
    male_sample_count = male_sample['Sex'].value_counts()
    female_sample = df[df['Sex'] == 0]
    female_sample_count = female_sample['Sex'].value_counts()
    disparity_sample_df = round((male_sample_count[1] -  female_sample_count[0])/ male_sample_count[1] * 100, 0)
    disparity_sample_df 
    return disparity_sample_df



def disparity_selection_percentage(df):
    male_selection = len(df[(df['Sex'] ==1)&(df['target'] ==1)]) / len(df[df["Sex"] == 1])
    female_selection = len(df[(df['Sex'] ==0)&(df['target'] ==1)]) / len(df[df["Sex"] == 0])
    disparity_selection_df = round((male_selection - female_selection)* 100, 0)
    return  disparity_selection_df


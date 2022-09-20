import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def vid(df, threshold):
    uncertainty = st.sidebar.radio("Type", ['Least Confidence Sampling','Margin of Confidence Sampling','Entropy'])

    df = df[df[uncertainty] > threshold]

    results = df['Label'].value_counts().sort_index()
    Count = results.values
    Class = results.index.values
    df_terbaru = pd.DataFrame(data = Count, index= Class,
                  columns = ['Count'])
    df_terbaru.reset_index(inplace=True)
    df_terbaru.rename(columns = {'index':'Class'}, inplace = True)
    df_terbaru['Class'] = df_terbaru['Class'].replace([0,1,2,3,4,5],['Backpack', 'Bed', 'Chair', 'Couch', 'Laptop', 'Table'])
    
    fig = px.bar(df_terbaru, x='Class', y='Count', color='Class')
    # fig.show()

    return fig

#Main
threshold = 0.6

st.set_page_config(layout = "wide")

xls = pd.ExcelFile('Data.xlsx')
df = pd.read_excel(xls, 'Main')

choose = ['Main','Day1', 'Day2', 'Day3', 'Day4', 'Day5', 'Day6', 'Day7', 'Day8']
df_main = pd.read_excel(xls, choose[0])
df1 = pd.read_excel(xls, choose[1])
df2 = pd.read_excel(xls, choose[2])
df3 = pd.read_excel(xls, choose[3])
df4 = pd.read_excel(xls, choose[4])
df5 = pd.read_excel(xls, choose[5])
df6 = pd.read_excel(xls, choose[6])
df7 = pd.read_excel(xls, choose[7])
df8 = pd.read_excel(xls, choose[7])


st.header("Uncertainty Score")

page = st.sidebar.selectbox('Select page', choose)
threshold = st.sidebar.slider("Threshold", min_value=0.1, max_value=1.0, value=0.6)


if page == 'Main':
    day = ['Day1', 'Day2', 'Day3', 'Day4', 'Day5', 'Day6', 'Day7', 'Day8']
    least = df['Least Confidence Sampling']
    margin = df['Margin of Confidence Sampling']
    entropy = df['Entropy']
    f1_macro = df['f1_macro']
    accuracy = df['Accuracy']

    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=day, y=least, name="Least", mode="lines"))
    fig.add_trace(go.Scatter(x=day, y=margin, name="Margin", mode="lines"))
    fig.add_trace(go.Scatter(x=day, y=entropy, name="Entropy", mode="lines"))
    fig.add_trace(go.Scatter(x=day, y=f1_macro, name="F1-Macro", mode="lines"))
    fig.add_trace(go.Scatter(x=day, y=accuracy, name="F1-Accuracy", mode="lines"))


    fig.update_layout(
        title="Uncertainty Graph", xaxis_title="Day", yaxis_title="Score"
    )
    # fig.show()
    st.plotly_chart(fig, use_container_width=True)


if page == 'Day1':
    fig = vid(df1, threshold)
    st.plotly_chart(fig, use_container_width=True)

if page == 'Day2':
    fig = vid(df2, threshold)
    st.plotly_chart(fig, use_container_width=True)

if page == 'Day3':
    fig = vid(df3, threshold)
    st.plotly_chart(fig, use_container_width=True)

if page == 'Day4':
    fig = vid(df4, threshold)
    st.plotly_chart(fig, use_container_width=True)

if page == 'Day5':
    fig = vid(df5, threshold)
    st.plotly_chart(fig, use_container_width=True)

if page == 'Day6':
    fig = vid(df6, threshold)
    st.plotly_chart(fig, use_container_width=True)

if page == 'Day7':
    fig = vid(df7, threshold)
    st.plotly_chart(fig, use_container_width=True)

if page == 'Day8':
    fig = vid(df8, threshold)
    st.plotly_chart(fig, use_container_width=True)


# if page == 'Country data':
#   ## Countries
#   clist = df['country'].unique()
#   country = st.selectbox("Select a country:",clist)
#   col1, col2 = st.columns(2)
#   fig = px.line(df[df['country'] == country], 
#     x = "year", y = "gdpPercap",title = "GDP per Capita")
 
#   col1.plotly_chart(fig,use_container_width = True)
#   fig = px.line(df[df['country'] == country], 
#     x = "year", y = "pop",title = "Population Growth")
  
#   col2.plotly_chart(fig,use_container_width = True)
# else:
#   ## Continents
#   contlist = df['continent'].unique()
 
#   continent = st.selectbox("Select a continent:",contlist)
#   col1,col2 = st.columns(2)
#   fig = px.line(df[df['continent'] == continent], 
#     x = "year", y = "gdpPercap",
#     title = "GDP per Capita",color = 'country')
  
#   col1.plotly_chart(fig)
#   fig = px.line(df[df['continent'] == continent], 
#     x = "year", y = "pop",
#     title = "Population",color = 'country')
  
#   col2.plotly_chart(fig, use_container_width = True)
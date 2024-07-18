import matplotlib.pyplot as plt
plt.hist(df['valence'],bins=20)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of valence')
plt.show()
plt.bar(df['key'].value_counts().index,
        df['key'].value_counts().values)
plt.xlabel('values')
plt.ylabel('Frequency')
plt.title('Bar chart of key')
plt.show()
plt.scattwe(df['popularity'],df['tempo'])
plt.xlabel('popularity')
plt.ylabel('tempo')
plt.title('Scatter Plot of popularity vs tempo')
plt.show()
import seaborn as sns
sns.boxplot(x='key',y='energy',data=data)
plt.xlabel('key')
plt.ylabel('energy')
plt.title('Box Plot of key and energy')
plt.show()
sns.pairplot(data)
plt.title('pair plot of numerical values')
plt.show()
import plotly.express as px
fig = px.scatter(data, x='liveness',y ='loudness',hover_data=['mode'])
fig.show()
import dash_core_components as dcc 
import dash_html_components as html 
app = dash.Dash(_name_) app.layout 
= html.Div([ dcc.Graph( 
id='interactive-plot', figure={ 'data': [ 
{'x': data['feature1'], 'y': data['feature2'], 'mode': 'markers', 'type': 
'scatter'} 
], 
'layout': { 
'title': 'Interactive Scatter Plot', 
'xaxis': {'title': 'Feature 1'}, 
'yaxis': {'title': 'Feature 2'} 
} 
} 
) 
]) 
if _name_ == '_main_': 
app.run_server(debug=True)
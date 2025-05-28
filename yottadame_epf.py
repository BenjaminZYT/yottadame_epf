import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

# Load CSVs (ensure these are present in the same directory)
CONV_full = pd.read_csv("conventional_fullfeatureset.csv", index_col=0)
CONV_full_trunc = CONV_full.dropna()
CONV_full_trunc_lagged = pd.read_csv("conventional_fullfeatureset_lagged.csv", index_col=0)

# Pie chart: EPF Asset Allocation
allocation_df = pd.DataFrame({
    'Asset Class': ['Fixed Income', 'Equities', 'Real Estate & Infrastructure', 'Money Market'],
    'Allocation (%)': [48, 40, 8, 4]
})
fig_pie = px.pie(allocation_df, names='Asset Class', values='Allocation (%)',
                 title='EPF Malaysia – Strategic Asset Allocation (Approx.)')

# Line chart: Simpanan Konvensional
fig_simpanan = px.line(CONV_full.reset_index(), x='Year', y='Simpanan Konvensional',
                       title='Simpanan Konvensional Over Time')
fig_simpanan.update_layout(xaxis_tickangle=-45, xaxis=dict(dtick=2))

# Scatter with regression: GDP vs Simpanan Konvensional
fig_gdp = px.scatter(CONV_full, x='gdp', y='Simpanan Konvensional', trendline='ols',
                     title='GDP vs Simpanan Konvensional')

# Correlation heatmap using seaborn
corr_matrix = CONV_full_trunc_lagged.corr()
sns.set(font_scale=1.0)
clustermap = sns.clustermap(
    corr_matrix,
    method='ward',
    cmap='coolwarm',
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    figsize=(7, 7)
)
buf = io.BytesIO()
plt.savefig(buf, format="png", bbox_inches='tight')
buf.seek(0)
encoded = base64.b64encode(buf.read()).decode("utf-8")
buf.close()
plt.close()
img_corr_clustermap = f"data:image/png;base64,{encoded}"

# Additional figures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

GE_years = {2004, 2008, 2013, 2018, 2022}

def add_GE_year_column(df):
    df = df.copy()
    df['GE_Year'] = df.index.to_series().apply(lambda year: 1 if year in GE_years else 0)
    return df

selected_features = ['usd_full_range', 'Investment', 'GE_Year']
target = 'Simpanan Konvensional'

train_data = CONV_full_trunc_lagged[[*selected_features[:-1], target]].dropna()
train_data = add_GE_year_column(train_data)
X_train = train_data.loc[2004:2019, selected_features]
y_train = train_data.loc[2004:2019, target]

test_data = train_data.loc[2020:2023, selected_features]
y_test_actual = train_data.loc[2020:2023, target]

X_2024 = CONV_full_trunc.loc[[2024], selected_features[:-1]].copy()
X_2024['GE_Year'] = 0
y_2024_actual = 6.30

X_test_full = pd.concat([test_data, X_2024])
y_test_full = pd.concat([y_test_actual, pd.Series([y_2024_actual], index=[2024])])

Model_GE = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])
Model_GE.fit(X_train, y_train)

y_pred = Model_GE.predict(X_test_full)
mse = mean_squared_error(y_test_full, y_pred)
mae = mean_absolute_error(y_test_full, y_pred)
r2 = r2_score(y_test_full, y_pred)

results = pd.DataFrame({
    'Actual': y_test_full,
    'Predicted': y_pred
}, index=y_test_full.index)
results.index = results.index + 1

# Create line plot of predictions
results_plot = results.reset_index().rename(columns={'index': 'Year'})
results_long = results_plot.melt(id_vars='Year', value_vars=['Actual', 'Predicted'],
                                 var_name='Type', value_name='Dividend')
fig_results = px.line(results_long, x='Year', y='Dividend', color='Type',
                      markers=True,
                      title='Actual vs Predicted EPF Dividend (Feature year → Target year +1)',
                      labels={'Dividend': 'Dividend (%)', 'Year': 'Year', 'Type': ''})
fig_results.update_layout(
    xaxis=dict(
        tickmode='linear',
        tick0=results_plot['Year'].min(),
        dtick=1
    ),
    template='plotly_white'
)

# App
app = dash.Dash(__name__)
app.title = "EPF Dividend Dashboard"

app.layout = html.Div([
    html.H1("\U0001F3AF Predicting EPF Dividend Rates", style={'marginTop': '2rem'}),

    html.Hr(),
    html.H3("EPF Asset Allocation (2024)"),
    dcc.Markdown("Breakdown of asset classes managed by EPF."),
    dcc.Graph(figure=fig_pie),

    html.Hr(),
    html.H3("Simpanan Konvensional Trends"),
    dcc.Markdown("Visual timeline of Simpanan Konvensional values over the years."),
    dcc.Graph(figure=fig_simpanan),

    html.Hr(),
    html.H3("Correlation Structure of Features"),
    dcc.Markdown("The following clustered heatmap shows the correlation between all features used in the model. It helps visualize which variables move together and may influence multicollinearity."),
    html.Img(src=img_corr_clustermap, style={"maxWidth": "100%", "height": "auto"}),

    html.Div([
    dcc.Markdown('''
    ### 📘 Feature Name Reference

    For clarity, here’s what each feature name refers to in the context of this model:

    - **opr_rate** → *OPR Rate* — representative figure chosen for the year (e.g., the most frequent value).
    - **usd_full_range** → *USD in MYR* — annual range of the USD/MYR exchange rate (max - min).
    - **klci_full_range** → *KLCI Index* — annual high-low range of the Bursa Malaysia index.
    - **snp_full_range** → *S&P Index* — annual high-low range of the S&P 500 index.
    - **Investment** → *Investment Amount (RM)* — total annual investment income reported by KWSP/EPF.
    - **gdp** → *Malaysia's GDP (RM)* — percentage growth of Malaysia's Gross Domestic Product.
    - **Simpanan Konvensional** → *Conventional EPF Dividend Rate (%)* — the model’s target variable.
    ''')
    ]),


    html.Hr(),
    html.H3("Predictions vs Actual (2021–2025)"),
    html.Div([
        html.Table([
            html.Thead([
                html.Tr([html.Th("Year")] + [html.Th(col) for col in results.columns])
            ]),
            html.Tbody([
                html.Tr([html.Td(f"{val:.2f}" if isinstance(val, float) else val) for val in row])
                for row in results.reset_index().itertuples(index=False)
                if isinstance(row, tuple)
                for row in [(row[0], *row[1:])]
            ])
        ], style={
            'margin': 'auto',
            'borderCollapse': 'collapse',
            'border': '1px solid #ddd',
            'textAlign': 'center',
            'width': '60%'
        })
    ]),

    html.H4("\U0001F4CA Model Evaluation Summary"),
    html.Div([
        html.Table([
            html.Thead([
                html.Tr([html.Th("Metric"), html.Th("Value")])
            ]),
            html.Tbody([
                html.Tr([html.Td("Mean Squared Error (MSE)"), html.Td("0.2352")]),
                html.Tr([html.Td("Mean Absolute Error (MAE)"), html.Td("0.3913")]),
                html.Tr([html.Td("R² Score"), html.Td("-0.4308")])
            ])
        ], style={
            'margin': 'auto',
            'borderCollapse': 'collapse',
            'border': '1px solid #ddd',
            'textAlign': 'center',
            'width': '40%'
        })
    ], style={'marginBottom': '2rem'}),

    dcc.Graph(figure=fig_results),

    dcc.Markdown('''
        We invite readers and fellow analysts to **improve upon this model**—whether by incorporating new data sources, exploring nonlinear relationships, or introducing better feature engineering. This project is meant as a starting point for deeper, collaborative modeling work.
    ''', style={'marginBottom': '4rem'})
])

app = dash.Dash(__name__)
app.title = "EPF Dividend Dashboard"
server = app.server

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "EPF Dividend Dashboard"
server = app.server

# Data for EPF asset allocation
allocation_data = {
    'Asset Class': ['Fixed Income', 'Equities', 'Real Estate & Infrastructure', 'Money Market'],
    'Allocation (%)': [48, 40, 8, 4]
}
allocation_df = pd.DataFrame(allocation_data)

# Pie chart
fig_allocation = px.pie(allocation_df, names='Asset Class', values='Allocation (%)',
                        title='EPF Malaysia – Strategic Asset Allocation (Approx.)')

# Load historical dividend data
CONV_full = pd.read_csv("conventional_fullfeatureset.csv", index_col=0)

# Line plot for Simpanan Konvensional
fig_dividend = px.line(CONV_full, x=CONV_full.index, y='Simpanan Konvensional',
                       title='EPF Simpanan Konvensional by Year',
                       labels={'x': 'Year', 'Simpanan Konvensional': 'Dividend (%)'},
                       markers=True)
fig_dividend.update_layout(
    xaxis=dict(tickmode='linear', tick0=int(CONV_full.index[0]), dtick=2, tickangle=45),
    template='plotly_white'
)

# Correlation heatmap
corr_matrix = CONV_full.corr()
sns.set(font_scale=1.0)
clustermap = sns.clustermap(
    corr_matrix, method='ward', cmap='coolwarm', annot=True,
    fmt=".2f", linewidths=0.5, figsize=(7, 7)
)
buf = BytesIO()
plt.savefig(buf, format="png")
buf.seek(0)
encoded_heatmap = base64.b64encode(buf.read()).decode("utf-8")
plt.close()

# Layout
app.layout = html.Div([
    html.H1("🎯 Predicting EPF Dividend Rates"),

    html.P("The Employees Provident Fund (EPF) is Malaysia’s national retirement savings scheme for private and non-pensionable public sector employees. "
	       "The EPF dividend rate is a highly anticipated figure among the Malaysian middle class each year. "
	       "As a benchmark of retirement savings growth, it directly impacts millions of contributors nationwide. "
           "Similar in spirit to the 401(k) system in the United States, it serves as a mandatory, "
	       "long-term savings vehicle designed to provide financial security after retirement. "
	       "However, unlike the 401(k), which is typically employer-managed and investment-choice driven, the EPF is centrally administered by a statutory body "
           "and invests collectively on behalf of all contributors. It guarantees a minimum annual dividend and covers a broader portion of the population by law."),
    html.P("In 2025, the EPF conventional dividend rate was notably high—bringing a wave of optimism and satisfaction among Malaysians."),
    html.P("In this notebook, we build a multiple linear regression model to predict the annual EPF conventional dividend rates, "
           "drawing insights from historical data and economic indicators. The goal is to understand the key drivers behind the dividend rate "
           "and to forecast future trends with data-driven confidence."),
    html.P("According to the official EPF website, the breakdown of assets held by EPF's investment arm in 2024 is as follows:"),

    dcc.Graph(figure=fig_allocation),
    html.Br(), html.Hr(),

    dcc.Graph(figure=fig_dividend),

    html.Br(), html.H2("Feature Correlation Heatmap"),

    html.H2("Feature Selection via Correlation Analysis"),
    html.P("Before building a regression model, it is crucial to understand how different variables relate to one another. "
           "For decision-makers, think of this like identifying which signals truly matter before making a financial call. "
           "If two features are highly similar (or unrelated), including both might either add noise or unnecessarily complicate the analysis. "
           "Correlation analysis helps us be more selective, prioritizing clarity and relevance in our model."),
    html.P("To refine our model and avoid redundancy or noise from weak predictors, we conducted a correlation analysis using a clustered heatmap (see above). "
           "The heatmap visualizes Pearson correlation coefficients between all numerical features, including our target variable: EPF Conventional Dividend Rate (Simpanan Konvensional)."),
    html.P("The analysis revealed several insights:"),
    html.Ul([
        html.Li("Investment Income (Investment) showed a strong positive correlation with Simpanan Konvensional, as expected—this feature directly reflects EPF’s earnings."),
        html.Li("USD/MYR Annual Exchange Rate Range (usd_full_range) also showed a meaningful positive correlation."),
        html.Li("We included General Election Year as a qualitative feature."),
        html.Li("GDP Growth (gdp) and S&P 500 Annual Range (snp_full_range) displayed moderate to weak correlations."),
        html.Li("Overnight Policy Rate (opr_rate) was excluded due to weak correlation and poor model contribution.")
    ]),
    html.Blockquote("Final Selected Features:\n"
                    "- Investment — Total investment income declared by EPF\n"
                    "- usd_full_range — Annual USD/MYR exchange rate range\n"
                    "- GE Year — General election year (Yes/No) dummy\n"
                    "- Simpanan Konvensional — Target variable"),
    html.P("This approach strikes a balance between predictive power and model simplicity, helping to avoid overfitting while retaining key explanatory variables."),
    html.P("The following heatmap reveals the correlations among all numeric features, offering insights into multicollinearity and potential drivers of the EPF dividend rate."),
    html.Img(src=f'data:image/png;base64,{encoded_heatmap}', style={'maxWidth': '100%'}),

    html.Br(), html.H2("Feature Overview"), html.Hr(),
    html.P("When modeling EPF's annual conventional dividend rate, one helpful way to think about the problem is through the lens of investment performance drivers."),
    html.P("A linear regression model offers a transparent, interpretable framework."),
    html.P("However, not all relevant data are publicly available."),

    html.H2("Feature Categories"),
    html.H3("1. Macroeconomic Indicators"),
    dcc.Markdown('''
| Feature | Description |
|--------|-------------|
| Malaysia GDP Growth (%) | National economic performance; influences earnings of domestic investments. |
| Malaysia Inflation Rate (%) | Affects real return; EPF aims to beat inflation over time. |
| Overnight Policy Rate (OPR) | Influences bond yields and monetary policy stance. |
| Exchange Rates (USD/GBP/EUR/JPY to MYR) | Affects returns from foreign investments once converted to MYR. |
| KLCI Index Annual Return (%) | Proxy for local equity performance. |
| MSCI World / S&P 500 Return (%) | Reflects global equity market performance. |
| Crude Oil Prices (Brent, USD/barrel) | Key driver of Malaysia’s economic health. |
| Political Party in Power | May influence fiscal policies and market confidence. |
| Trade Surplus/Deficit | Indicator of economic health and external demand. |
'''),

    html.H3("2. EPF Portfolio Indicators (If Available)"),
    dcc.Markdown('''
| Feature | Description |
|--------|-------------|
| Total Investment Income (RM) | Declared income from EPF’s investment activities. |
| Equity Allocation (%) | Higher equity exposure often implies higher expected returns. |
| Fixed Income Allocation (%) | Reflects stability and predictable income. |
| Real Estate & Infrastructure Allocation (%) | Long-term cash flow and inflation hedges. |
| Dividend Income (RM) | Earnings from equity holdings. |
| MGS Yield (%) | Benchmark for sovereign bond returns. |
'''),

    html.H3("3. Policy & Behavioral Factors"),
    dcc.Markdown('''
| Feature | Description |
|--------|-------------|
| EPF Statutory Contribution Rate (%) | Influences fund inflow. |
| Withdrawal Programs (Dummy Variable) | e.g., i-Lestari, i-Citra. |
| Net Contribution Inflow (RM) | Captures liquidity pressure or availability. |
| General Election Year | Market/policy uncertainty. |
| Political Party in Executive Power | May affect investment priorities. |
'''),

    html.H3("4. Lagged Variables"),
    html.P("Lagged variables capture delayed effects over time:"),
    html.Ul([
        html.Li("Previous Year’s EPF Dividend Rate"),
        html.Li("Previous Year’s KLCI or MSCI Returns"),
        html.Li("Previous Year’s GDP Growth or Inflation")
    ]),

    html.Br(), html.H2("Data Availability and Feature Selection"), html.Hr(),
    html.P("While there are many possible features, data availability is a major constraint."),
    html.P("Final feature set includes:"),
    html.Ul([
        html.Li("EPF Conventional Dividend Rate (%): Target variable."),
        html.Li("Overnight Policy Rate (OPR): Mode-based annual summary."),
        html.Li("USD/MYR Exchange Rate Range: Annual volatility range."),
        html.Li("KLCI Index Annual Range: Market movement range."),
        html.Li("S&P 500 Index Annual Range: Global market volatility."),
        html.Li("Malaysia GDP Growth Rate (%): Macroeconomic proxy."),
        html.Li("General Election Year: Binary indicator.")
    ]),
    html.P("Note: Annual ranges consistently outperformed Q4-based features."),
    html.P("This curated set balances relevance, quality, and availability."),

    html.H2("Model: Linear Regression with General Election Year"),

    html.P("To predict the EPF dividend rate, we use a standard linear regression model."),

    html.H3("Selected Features"),
    html.Ul([
        html.Li("USD/MYR Exchange Rate Annual Range (usd_full_range): Captures volatility in the foreign exchange market and its potential impact on international investments."),
        html.Li("EPF Total Investment Income (Investment): Represents the total income declared from EPF’s investment operations—arguably the most direct driver of the declared dividend rate."),
        html.Li("General Election Year (Dummy Variable: GE_Year): Encodes whether a general election occurred in a given year, capturing the potential influence of political uncertainty or fiscal changes.")
    ]),

    html.H3("Train-Test Split Strategy"),
    html.P("Training Period: Features from 2004–2020"),
    html.P("Testing Period: Features from 2021–2024"),

    html.H3("Model Performance"),

    html.Div([
        html.H4("Linear Regression (Train 2004–2020, Test 2021–2024)"),
        html.P("Mean Squared Error (MSE):  0.3955"),
        html.P("Mean Absolute Error (MAE): 0.5254"),
        html.P("R² Score: -1.5035"),
        html.Br(),
        html.Table([
            html.Thead(html.Tr([
                html.Th("Year"), html.Th("Actual"), html.Th("Predicted")
            ])),
            html.Tbody([
                html.Tr([html.Td("2021"), html.Td("6.10"), html.Td("6.161659")]),
                html.Tr([html.Td("2022"), html.Td("5.35"), html.Td("6.058486")]),
                html.Tr([html.Td("2023"), html.Td("5.50"), html.Td("6.473715")]),
                html.Tr([html.Td("2024"), html.Td("6.30"), html.Td("6.657827")])
            ])
        ], style={"margin": "0 auto", "border": "1px solid #ccc", "borderCollapse": "collapse", "width": "60%"}),

        dcc.Graph(figure=px.line(
            pd.DataFrame({
                'Year': [2021, 2022, 2023, 2024]*2,
                'Dividend': [6.10, 5.35, 5.50, 6.30, 6.161659, 6.058486, 6.473715, 6.657827],
                'Type': ['Actual']*4 + ['Predicted']*4
            }),
            x='Year',
            y='Dividend',
            color='Type',
            markers=True,
            title='Actual vs Predicted EPF Dividend',
            labels={'Dividend': 'Dividend (%)', 'Year': 'Year', 'Type': ''}
        ).update_layout(
            xaxis=dict(tickmode='linear', tick0=2021, dtick=1),
            template='plotly_white'
        ))
    ], style={"textAlign": "center"}),

    html.Br(),
    html.H2("Forecast for 2025", style={"textAlign": "center"}),

    html.H3("Predicted EPF Conventional Dividend Rate for 2025: 6.51%",
            style={"textAlign": "center", "color": "#1F618D"}),

    html.Br(),

    html.P(
        "Important: Note that this is by no means an investment advice. "
        "This figure is derived from a model based on publicly available data.",
        style={"textAlign": "left", "fontStyle": "bold"}
    ),

    html.P(
        "We invite readers and fellow analysts to improve upon this model—"
        "whether by incorporating new data sources, exploring nonlinear relationships, "
        "or introducing better feature engineering. This project is meant as a starting point "
        "for deeper, collaborative modeling work.",
        style={"textAlign": "left", "fontStyle": "italic"}
    )

])

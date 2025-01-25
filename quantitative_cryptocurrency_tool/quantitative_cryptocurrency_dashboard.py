import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
from dash.dependencies import State
import dash_bootstrap_components as dbc
from prophet import Prophet
from prophet.plot import plot_plotly

#Exchange connection
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'adjustForTimeDifference': True
    }
})

CRYPTO_PAIRS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
    'SOL/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'LUNA/USDT',
    'MATIC/USDT', 'ATOM/USDT', 'ALGO/USDT', 'LTC/USDT', 'UNI/USDT'
]


def fetch_ohlcv(symbol, timeframe='1d', limit=1000):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()


def calculate_technical_indicators(df):
    #Indicators
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['close'], 14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = calculate_macd(df['close'])
    return df


def calculate_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def generate_forecast(df, periods=365):
    df_prophet = df.reset_index()[['timestamp', 'close']].rename(columns={'timestamp': 'ds', 'close': 'y'})
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.add_country_holidays(country_name='US')
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast, model


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

#Custom CSS
app.css.append_css({
    'external_url': 'https://codepen.io/juuso/pen/ZYzmjwO'  # Dash CSS
})

#Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Quantitative Analytics Platform",
                        className='text-center mb-4'),
                width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Parameters", className="card-title"),
                    dcc.Dropdown(
                        id='symbol-select',
                        options=[{'label': pair, 'value': pair}
                                 for pair in CRYPTO_PAIRS],
                        value='BTC/USDT',
                        className='mb-3',
                        style={
                            'backgroundColor': '#303030',
                            'color': 'black',
                            'border': '1px solid #555',
                        }
                    ),
                    dcc.Dropdown(
                        id='timeframe-select',
                        options=[
                            {'label': '1 Minute', 'value': '1m'},
                            {'label': '15 Minutes', 'value': '15m'},
                            {'label': '1 Hour', 'value': '1h'},
                            {'label': '4 Hours', 'value': '4h'},
                            {'label': '1 Day', 'value': '1d'}
                        ],
                        value='1d',
                        className='mb-3',
                        style={
                            'backgroundColor': '#303030',
                            'color': 'black',
                            'border': '1px solid #555',
                        }
                    ),
                    dbc.Button("Run Analysis",
                               id='run-button',
                               color="primary",
                               className="w-100")
                ])
            ], className='mt-3')
        ], width=2),
        dbc.Col([
            dbc.Row([
                dbc.Col(dcc.Graph(id='price-chart'), width=12,
                        className='p-0'),
                dbc.Col(dcc.Graph(id='volume-chart'), width=12,
                        className='p-0'),
                dbc.Col(dcc.Graph(id='forecast-chart'), width=12,
                        className='p-0')
            ], className='g-0')
        ], width=10)
    ]),
    dcc.Interval(
        id='interval-component',
        interval=60 * 1000,
        n_intervals=0
    )
], fluid=True, style={'padding': '0'})


@app.callback(
    [Output('price-chart', 'figure'),
     Output('volume-chart', 'figure'),
     Output('forecast-chart', 'figure')],
    [Input('run-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')],
    [State('symbol-select', 'value'),
     State('timeframe-select', 'value')]
)
def update_charts(n_clicks, n_intervals, symbol, timeframe):
    df = fetch_ohlcv(symbol, timeframe)
    if df.empty:
        return go.Figure(), go.Figure(), go.Figure()

    df = calculate_technical_indicators(df)

    # Price charts
    price_fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              vertical_spacing=0.1,  # Increased vertical spacing
                              row_heights=[0.7, 0.3])

    price_fig.add_trace(go.Candlestick(x=df.index,
                                       open=df['open'],
                                       high=df['high'],
                                       low=df['low'],
                                       close=df['close'],
                                       name='Price'), row=1, col=1)

    price_fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'],
                                   line=dict(color='orange', width=1),
                                   name='SMA 20'), row=1, col=1)

    price_fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'],
                                   line=dict(color='blue', width=1),
                                   name='SMA 50'), row=1, col=1)

    # MACD
    price_fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'],
                               name='MACD Histogram'), row=2, col=1)
    price_fig.add_trace(go.Scatter(x=df.index, y=df['MACD'],
                                   line=dict(color='green', width=1),
                                   name='MACD'), row=2, col=1)
    price_fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'],
                                   line=dict(color='red', width=1),
                                   name='Signal Line'), row=2, col=1)

    price_fig.update_layout(
        template='plotly_dark',
        margin=dict(l=0, r=0, t=50, b=80),  # Increased bottom margin
        height=500,  # Increased height
        title=f'{symbol} Price Analysis',
        title_x=0.5,
        xaxis=dict(
            rangeslider=dict(
                visible=True,
                thickness=0.1,  # Adjust slider thickness if needed
            ),
        )
    )

    #Volume chart
    volume_fig = go.Figure(go.Bar(x=df.index, y=df['volume'],
                                  marker_color=np.where(
                                      df['close'] > df['open'],
                                      'green', 'red')))
    volume_fig.update_layout(
        template='plotly_dark',
        margin=dict(l=0, r=0, t=50, b=0),
        height=200,
        title=f'{symbol} Volume',
        title_x=0.5
    )

    #Forecast chart
    forecast, model = generate_forecast(df)
    forecast_fig = plot_plotly(model, forecast)
    forecast_fig.update_layout(
        template='plotly_dark',
        margin=dict(l=0, r=0, t=50, b=0),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        title=f'{symbol} Price Forecast',
        title_x=0.5
    )

    forecast_fig.update_layout(
        modebar=dict(bgcolor='rgba(0,0,0,0)'),
        title_x=0.5,
        title_y=0.95
    )

    return price_fig, volume_fig, forecast_fig


if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
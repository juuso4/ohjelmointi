import os
import sys
import time
import yfinance as yf
import pandas as pd
import numpy as np
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns
import mplcursors
import requests
from datetime import datetime, timedelta
from matplotlib import gridspec
from matplotlib.dates import DateFormatter, WeekdayLocator, num2date
from matplotlib.ticker import FuncFormatter


class EnhancedFinancialTerminal:
    def __init__(self):
        self.console = Console()
        self.layout = Layout()
        self.asset_data = None
        self.selected_asset = "BTC-USD"
        self.timeframe_days = 30
        self.interval = '1d'
        self.last_fetch = 0
        self.rate_limit = 2
        self.moving_averages = [20, 50, 200]
        self.forecast_confidence = 0.95
        self.show_all_indices = False

        # Country Index Mapping
        self.country_indices = {
            'US': '^GSPC',  # S&P 500
            'NASDAQ': '^IXIC',  # NASDAQ Composite
            'Japan': '^N225',  # Nikkei 225
            'UK': '^FTSE',  # FTSE 100
            'Finland': '^OMXH25',  # OMX Helsinki 25
        }

        if os.name == 'nt':
            from ctypes import windll
            windll.kernel32.SetConsoleMode(windll.kernel32.GetStdHandle(-11), 7)

        self.initialize_layout()
        self.console.print("[yellow]Initializing Aurora Terminal...[/]")

    def initialize_layout(self):
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        self.layout['header'].update(Panel("Loading...", style="blue"))
        self.layout['main'].update(Panel("Initializing...", style="yellow"))
        self.layout['footer'].update(
            Panel("Aurora Terminal v1.1 | Commands: q=quit, a=asset, t=timeframe, f=forecast, tv/pv=charts, i=indices",
                  style="green"))

    def fetch_data(self):
        try:
            if time.time() - self.last_fetch < self.rate_limit:
                time.sleep(self.rate_limit - (time.time() - self.last_fetch))

            ticker = yf.Ticker(self.selected_asset)
            df = ticker.history(period=f"{self.timeframe_days}d", interval=self.interval)

            if df.empty:
                raise ValueError("No data available for selected asset")

            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.columns = [col.lower() for col in df.columns]
            df = df.ffill().bfill()

            self.last_fetch = time.time()
            return df

        except Exception as e:
            self.show_error(f"Data Error: {str(e)}")
            return None

    def get_current_price(self):
        try:
            if time.time() - self.last_fetch < self.rate_limit:
                return self.asset_data['close'].iloc[-1] if self.asset_data is not None else 0

            ticker = yf.Ticker(self.selected_asset)
            data = ticker.history(period='1d', interval='1m')
            return data['Close'].iloc[-1] if not data.empty else self.asset_data['close'].iloc[-1]
        except Exception:
            return self.asset_data['close'].iloc[-1] if self.asset_data is not None else 0

    def change_asset(self):
        new_asset = input("Enter asset symbol (e.g. BTC-USD, AAPL): ").strip().upper()
        self.selected_asset = new_asset
        self.asset_data = None
        self.console.print(f"[green]Switched to {new_asset}[/]")
        time.sleep(0.5)

    def change_timeframe(self):
        try:
            new_timeframe = int(input("Enter new timeframe (7-365 days): "))
            if 7 <= new_timeframe <= 365:
                self.timeframe_days = new_timeframe
                self.asset_data = None
                self.console.print(f"[green]Timeframe set to {new_timeframe} days[/]")
                time.sleep(0.5)
            else:
                self.show_error("Must be between 7-365 days")
                time.sleep(1)
        except ValueError:
            self.show_error("Invalid number entered")
            time.sleep(1)

    def calculate_technical_indicators(self):
        if self.asset_data is None:
            return

        #RSI
        delta = self.asset_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.asset_data['rsi'] = 100 - (100 / (1 + rs))

        #MACD
        exp12 = self.asset_data['close'].ewm(span=12, adjust=False).mean()
        exp26 = self.asset_data['close'].ewm(span=26, adjust=False).mean()
        self.asset_data['macd'] = exp12 - exp26
        self.asset_data['signal'] = self.asset_data['macd'].ewm(span=9, adjust=False).mean()

        #Moving Averages and Bollinger Bands
        for ma in self.moving_averages:
            self.asset_data[f'sma_{ma}'] = self.asset_data['close'].rolling(window=ma).mean()
            std = self.asset_data['close'].rolling(window=ma).std()
            self.asset_data[f'upper_band_{ma}'] = self.asset_data[f'sma_{ma}'] + (std * 2)
            self.asset_data[f'lower_band_{ma}'] = self.asset_data[f'sma_{ma}'] - (std * 2)

        #VWAP
        cumulative_tp = (self.asset_data['high'] + self.asset_data['low'] + self.asset_data['close']).div(3) * \
                        self.asset_data['volume']
        cumulative_vol = self.asset_data['volume'].cumsum()
        self.asset_data['vwap'] = cumulative_tp.cumsum() / cumulative_vol

    def volume_analysis(self):
        try:
            return {
                'volume_change': self.asset_data['volume'].pct_change().iloc[-1],
                'volume_avg_7d': self.asset_data['volume'].rolling(7).mean().iloc[-1]
            }
        except Exception:
            return None

    def forecast_price(self, method='arima', days=7):
        try:
            if self.asset_data is None or len(self.asset_data) < 30:
                raise ValueError("Need at least 30 days of data for forecasting")

            prices = self.asset_data['close'].values
            if method == 'linear':
                model = LinearRegression()
                X = np.arange(len(prices)).reshape(-1, 1)
                model.fit(X, prices)
                return model.predict(np.arange(len(prices), len(prices) + days).reshape(-1, 1))
            elif method == 'arima':
                model = ARIMA(prices, order=(5, 1, 0))
                model_fit = model.fit()
                forecast = model_fit.get_forecast(steps=days)
                return forecast.predicted_mean, forecast.conf_int(alpha=1 - self.forecast_confidence)
            return None
        except Exception as e:
            self.show_error(f"Forecast Error: {str(e)}")
            return None

    def _get_hover_text(self, sel, df):
        #Generate hover tooltip text from cursor position
        try:
            date = num2date(sel.target[0]).replace(tzinfo=None)
            idx = df.index.get_indexer([date], method='nearest')[0]
            nearest_date = df.index[idx]
            row = df.loc[nearest_date]

            return (
                f"Date: {nearest_date.strftime('%Y-%m-%d')}\n"
                f"Open: {row['open']:.2f}\n"
                f"High: {row['high']:.2f}\n"
                f"Low: {row['low']:.2f}\n"
                f"Close: {row['close']:.2f}\n"
                f"Volume: {row['volume']:,.0f}"
            )
        except Exception as e:
            return f"Error: {str(e)}"

    def draw_chart(self, chart_type='technical'):
        try:
            plt.close('all')
            sns.set_style('darkgrid')
            plt.rcParams.update({
                'axes.titlecolor': 'white',
                'axes.labelcolor': 'white',
                'xtick.color': 'white',
                'ytick.color': 'white',
                'figure.facecolor': '#1a1a1a',
                'axes.facecolor': '#1a1a1a'
            })

            if self.asset_data is None:
                raise ValueError("No data to visualize")

            df = self.asset_data.copy()
            df.index = pd.to_datetime(df.index)

            if chart_type == 'technical':
                fig = plt.figure(figsize=(14, 12))
                gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1, 1])
                ax1 = plt.subplot(gs[:2])
                ax2 = plt.subplot(gs[2], sharex=ax1)
                ax3 = plt.subplot(gs[3], sharex=ax1)
            else:
                fig = plt.figure(figsize=(14, 8))
                gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
                ax1 = plt.subplot(gs[0])
                ax2 = plt.subplot(gs[1], sharex=ax1)

            #Main Price Chart
            for date, row in df.iterrows():
                color = '#2ecc71' if row['close'] >= row['open'] else '#e74c3c'
                ax1.vlines(date, row['low'], row['high'], color=color, linewidth=1)
                ax1.plot([date, date], [row['open'], row['close']],
                         color=color, linewidth=3, solid_capstyle='butt')

            for ma in self.moving_averages:
                ax1.plot(df.index, df[f'sma_{ma}'],
                         label=f'{ma} SMA', linewidth=1.5, alpha=0.7)

            ax1.plot(df.index, df['vwap'], label='VWAP', color='#f1c40f', linewidth=1.5)
            ax1.set_title(f"{self.selected_asset} {'Technical' if chart_type == 'technical' else 'Price'} Analysis",
                          pad=20, fontsize=14, weight='bold')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.2f}'))
            plt.setp(ax1.get_xticklabels(), visible=False)

            #Volume Chart
            ax2.bar(df.index, df['volume'],
                    color=['#2ecc71' if close >= open_ else '#e74c3c'
                           for close, open_ in zip(df['close'], df['open'])],
                    width=0.8, alpha=0.4)
            ax2.set_ylabel('Volume', labelpad=10)
            ax2.grid(True, alpha=0.3)
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x / 1e6:,.1f}M'))

            if chart_type == 'technical':
                plt.setp(ax2.get_xticklabels(), visible=False)

                #MACD
                ax3.plot(df.index, df['macd'], label='MACD', color='#2ecc71')
                ax3.plot(df.index, df['signal'], label='Signal', color='#e74c3c')
                ax3.bar(df.index, df['macd'] - df['signal'],
                        color=np.where((df['macd'] - df['signal']) >= 0, '#2ecc71', '#e74c3c'),
                        alpha=0.3)
                ax3.axhline(0, color='white', linestyle='--', linewidth=0.5)
                ax3.legend(loc='upper left')
                ax3.set_ylabel('MACD', labelpad=10)
                ax3.grid(True, alpha=0.3)

                #RSI
                ax4 = ax3.twinx()
                ax4.plot(df.index, df['rsi'], color='#9b59b6', label='RSI')
                ax4.fill_between(df.index, 30, 70, color='#2c3e50', alpha=0.1)
                ax4.axhline(30, color='#e74c3c', linestyle='--', linewidth=0.8)
                ax4.axhline(70, color='#e74c3c', linestyle='--', linewidth=0.8)
                ax4.set_ylabel('RSI', labelpad=10)
                ax4.set_ylim(0, 100)
                ax4.legend(loc='upper right')

            #Date formatting
            date_format = DateFormatter('%Y-%m-%d')
            (ax3 if chart_type == 'technical' else ax2).xaxis.set_major_formatter(date_format)
            (ax3 if chart_type == 'technical' else ax2).xaxis.set_major_locator(WeekdayLocator(byweekday=(0, 6)))
            plt.xticks(rotation=45, ha='right')

            #Add interactive cursor
            cursor = mplcursors.cursor(ax1, hover=True)
            cursor.connect("add", lambda sel: sel.annotation.set_text(
                self._get_hover_text(sel, df)
            ))

            plt.tight_layout()
            plt.subplots_adjust(hspace=0.05)
            plt.show()

        except Exception as e:
            self.show_error(f"Chart Error: {str(e)}")
            plt.close('all')

    def draw_forecast(self, forecast_data):

        try:
            plt.close('all')
            sns.set_style('darkgrid')
            plt.rcParams.update({
                'axes.titlecolor': 'white',
                'axes.labelcolor': 'white',
                'xtick.color': 'white',
                'ytick.color': 'white',
                'figure.facecolor': '#1a1a1a',
                'axes.facecolor': '#1a1a1a'
            })

            forecast, conf_int = forecast_data
            historical = self.asset_data['close'].values
            days = len(forecast)

            fig, ax = plt.subplots(figsize=(14, 7))

            #Plot historical
            ax.plot(np.arange(len(historical)), historical,
                    color='#3498db', linewidth=2, label='Historical')

            # Plot forecast
            ax.plot(np.arange(len(historical), len(historical) + days), forecast,
                    color='#e67e22', linewidth=2, linestyle='--', label='Forecast')

            #Confidence interval shading
            ax.fill_between(np.arange(len(historical), len(historical) + days),
                            conf_int[:, 0], conf_int[:, 1],
                            color='#e67e22', alpha=0.15, label='95% Confidence')

            ax.set_title(f"{self.selected_asset} Price Forecast",
                         pad=20, fontsize=14, weight='bold')
            ax.set_xlabel('Trading Days', labelpad=10)
            ax.set_ylabel('Price', labelpad=10)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.2f}'))
            ax.legend()
            ax.grid(True, alpha=0.3)

            #Annotate forecast points
            for i, (pred, lower, upper) in enumerate(zip(forecast, conf_int[:, 0], conf_int[:, 1])):
                ax.annotate(f'{pred:.2f}',
                            (len(historical) + i, pred),
                            textcoords="offset points",
                            xytext=(0, 10), ha='center',
                            color='#e67e22')

            plt.tight_layout()
            plt.show()

        except Exception as e:
            self.show_error(f"Forecast Visualization Error: {str(e)}")
            plt.close('all')

    def update_header(self):
        try:
            ticker = yf.Ticker(self.selected_asset)
            info = ticker.info
            current_price = self.get_current_price()

            header_text = Text.assemble(
                ("Aurora Terminal ", "bold cyan"),
                f"| {info.get('shortName', self.selected_asset)} ${current_price:.2f} ",
                f"| Market Cap: ${info.get('marketCap', 'N/A'):,.0f} ",
                f"| Volume: {self.asset_data['volume'].iloc[-1]:,.0f} " if self.asset_data is not None else "",
            )
            self.layout['header'].update(Panel(header_text, style="white"))
        except Exception:
            self.layout['header'].update(Panel("Real-time data unavailable", style="red"))

    def show_main_view(self):
        try:
            if self.asset_data is None:
                self.asset_data = self.fetch_data()
                if self.asset_data is not None:
                    self.calculate_technical_indicators()

            if self.asset_data is None or self.asset_data.empty:
                self.layout['main'].update(Panel("Failed to load data - Check symbol", style="red"))
                return

            #Create stats table
            stats_table = Table(box=box.SIMPLE, expand=True)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="magenta")

            current_price = self.asset_data['close'].iloc[-1]
            stats_table.add_row("Current Price", f"${current_price:.2f}")
            stats_table.add_row(f"{self.timeframe_days}d Change",
                                f"{(current_price / self.asset_data['close'].iloc[0] - 1) * 100:.2f}%")
            stats_table.add_row("RSI (14)", f"{self.asset_data['rsi'].iloc[-1]:.1f}")
            stats_table.add_row("MACD", f"{self.asset_data['macd'].iloc[-1]:.2f}")
            stats_table.add_row("Bollinger %",
                                f"{(current_price - self.asset_data['lower_band_20'].iloc[-1]) / (self.asset_data['upper_band_20'].iloc[-1] - self.asset_data['lower_band_20'].iloc[-1]) * 100:.1f}%")

            vol_analysis = self.volume_analysis()
            if vol_analysis:
                stats_table.add_row("Volume Change",
                                    f"{vol_analysis.get('volume_change', 0) * 100:.2f}%")
                stats_table.add_row("7d Avg Volume",
                                    f"{vol_analysis.get('volume_avg_7d', 0):,.0f}")

            #Create indices table with pagination
            indices_table = Table(box=box.SIMPLE, expand=True)
            indices_table.add_column("Country", style="cyan")
            indices_table.add_column("Index", style="magenta")
            indices_table.add_column("Price", style="green")
            indices_table.add_column("Change", style="yellow")

            #Get all indices data
            indices_data = []
            for country, ticker in self.country_indices.items():
                try:
                    data = yf.Ticker(ticker).history(period='1d')
                    if not data.empty:
                        price = data['Close'].iloc[-1]
                        change = (price / data['Close'].iloc[0] - 1) * 100
                        indices_data.append({
                            'country': country,
                            'ticker': ticker,
                            'price': price,
                            'change': change
                        })
                except Exception:
                    continue

            #Show all indices if requested, otherwise first 6
            if self.show_all_indices:
                for index in indices_data:
                    indices_table.add_row(
                        index['country'],
                        index['ticker'],
                        f"${index['price']:,.2f}",
                        f"{index['change']:+.2f}%"
                    )
            else:
                for index in indices_data[:6]:
                    indices_table.add_row(
                        index['country'],
                        index['ticker'],
                        f"${index['price']:,.2f}",
                        f"{index['change']:+.2f}%"
                    )
                if len(indices_data) > 6:
                    indices_table.add_row(
                        "...", "...", "...", "...",
                        style="italic"
                    )

            #Create indicator table
            indicator_table = Table(box=box.SIMPLE, expand=True)
            indicator_table.add_column("Indicator", style="green")
            indicator_table.add_column("Description", style="yellow")
            indicator_table.add_row("RSI", "30=Oversold, 70=Overbought")
            indicator_table.add_row("MACD", "Positive=Upward momentum")
            indicator_table.add_row("Bollinger %", "<20=Oversold, >80=Overbought")

            #Update layout
            grid = Table.grid(expand=True)
            grid.add_row(Panel(stats_table, title="Market Stats"))
            grid.add_row(Panel(indices_table, title="Global Indices"))
            grid.add_row(Panel(indicator_table, title="Indicator Guide"))

            self.layout['main'].update(grid)
            self.update_header()

        except Exception as e:
            self.show_error(f"Display Error: {str(e)}")

    def show_error(self, message):
        error_panel = Panel(Text(message, style="bold red"),
                            title="Error", border_style="red")
        self.layout['main'].update(error_panel)

    def run(self):
        try:
            while True:
                os.system('cls' if os.name == 'nt' else 'clear')
                self.show_main_view()
                self.console.print(self.layout)

                user_input = input("\nCommand (q=quit, a=asset, t=timeframe, f=forecast, "
                                   "tv/pv=charts, i=indices: ").lower()

                if user_input == 'q':
                    break
                elif user_input == 'a':
                    self.change_asset()
                elif user_input == 't':
                    self.change_timeframe()
                elif user_input == 'f':
                    forecast = self.forecast_price()
                    if forecast is not None:
                        self.draw_forecast(forecast)
                elif user_input in ('tv', 'pv'):
                    self.draw_chart(chart_type='technical' if user_input == 'tv' else 'price')
                elif user_input == 'i':
                    self.show_all_indices = not self.show_all_indices

        except Exception as e:
            self.console.print(f"[bold red]Fatal Error: {str(e)}[/]")
            import traceback
            traceback.print_exc()
        finally:
            input("\nPress Enter to exit...")


if __name__ == "__main__":
    terminal = EnhancedFinancialTerminal()
    terminal.run()
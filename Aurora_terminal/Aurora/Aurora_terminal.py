import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Back, Style, init
from termcolor import colored
import mplcursors
from scipy.stats import linregress
import os
import sys
import warnings
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

warnings.filterwarnings("ignore")
init(autoreset=True)

# Configuration
sns.set_theme(style="darkgrid")
sns.set_palette("husl")
pd.set_option('display.float_format', lambda x: '%.2f' % x)


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def get_ticker_data():
    clear_screen()
    print(colored("\n 🚀Aurora Terminal", "cyan", attrs=["bold"]))
    print(colored("=" * 50, "blue"))

    while True:
        ticker = input(colored("\nEnter ticker symbol (e.g., AAPL or BTC-USD): ", "yellow")).upper()
        if not ticker:
            print(colored("Please enter a valid ticker!", "red"))
            continue

        try:
            asset = yf.Ticker(ticker)
            hist = asset.history(period="1y")
            if hist.empty:
                raise ValueError
            return asset, hist
        except:
            print(colored(f"Invalid ticker or no data available for {ticker}! Try again.", "red"))


def display_basic_info(asset, hist):
    info = asset.info
    print(colored("\n📈 BASIC INFORMATION", "green", attrs=["bold"]))
    print(colored("-" * 50, "blue"))

    current_price = info.get('currentPrice', info.get('regularMarketPrice', hist['Close'].iloc[-1]))
    previous_close = info.get('previousClose', hist['Close'].iloc[-2] if len(hist) > 1 else 'N/A')

    basics = {
        'name': info.get('shortName', 'N/A'),
        'current_price': current_price,
        'previous_close': previous_close,
        'market_cap': info.get('marketCap', 'N/A'),
        'volume': info.get('volume', hist['Volume'].iloc[-1] if len(hist) > 1 else 'N/A'),
        '52w_high': info.get('fiftyTwoWeekHigh', 'N/A'),
        '52w_low': info.get('fiftyTwoWeekLow', 'N/A'),
        'currency': info.get('currency', 'USD')
    }

    for key, value in basics.items():
        print(colored(f"{key.replace('_', ' ').title():<15}: ", "cyan") + f"{value}")


def calculate_indicators(hist, benchmark_ticker="^GSPC"):
    print(colored("\n📊 TECHNICAL ANALYSIS SUMMARY", "green", attrs=["bold"]))
    print(colored("-" * 50, "blue"))

    # Moving Averages (SMA/EMA)
    hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
    hist['EMA_20'] = hist['Close'].ewm(span=20, adjust=False).mean()

    # Relative Strength Index (RSI)
    hist['RSI'] = ta.rsi(hist['Close'], length=14)

    # Moving Average Convergence Divergence (MACD)
    macd = ta.macd(hist['Close'], fast=12, slow=26, signal=9)
    hist['MACD'] = macd[f'MACD_12_26_9']
    hist['MACD_Signal'] = macd[f'MACDs_12_26_9']
    hist['MACD_Hist'] = macd[f'MACDh_12_26_9']

    # Bollinger Bands
    bbands_df = ta.bbands(hist['Close'], length=20)
    hist['Upper_BB'] = bbands_df[f'BBU_20_2.0']
    hist['Middle_BB'] = bbands_df[f'BBM_20_2.0']
    hist['Lower_BB'] = bbands_df[f'BBL_20_2.0']

    # Stochastic Oscillator
    stoch = ta.stoch(hist['High'], hist['Low'], hist['Close'], k=14, d=3)
    hist['STOCH_%K'] = stoch[f'STOCHk_14_3_3']
    hist['STOCH_%D'] = stoch[f'STOCHd_14_3_3']

    # Volatility (Standard Deviation and ATR)
    hist['Std_Dev'] = hist['Close'].rolling(window=20).std()
    hist['ATR'] = ta.atr(hist['High'], hist['Low'], hist['Close'], length=14)

    # Historical Volatility (Annualized)
    hist['Returns'] = hist['Close'].pct_change()
    hist['Hist_Volatility'] = hist['Returns'].rolling(window=20).std() * np.sqrt(252)

    # Alpha and Beta
    benchmark = yf.Ticker(benchmark_ticker).history(period="1y")['Close']
    benchmark_returns = benchmark.pct_change().dropna()
    asset_returns = hist['Close'].pct_change().dropna()
    common_dates = benchmark_returns.index.intersection(asset_returns.index)

    if len(common_dates) < 2:
        print(colored(f"Warning: Not enough common dates with {benchmark_ticker} to calculate Alpha and Beta.", "yellow"))
        alpha, beta = np.nan, np.nan
    else:
        beta, alpha, _, _, _ = linregress(benchmark_returns[common_dates], asset_returns[common_dates])

    sharpe_ratio = hist['Returns'].mean() / hist['Returns'].std() * np.sqrt(252)

    # Display
    print(colored(f"20D SMA: {hist['SMA_20'].iloc[-1]:.2f}", "cyan"))
    print(colored(f"20D EMA: {hist['EMA_20'].iloc[-1]:.2f}", "cyan"))
    print(colored(f"RSI (14): {hist['RSI'].iloc[-1]:.2f}", "cyan") +
          " - " + ("Overbought (>70)" if hist['RSI'].iloc[-1] > 70 else "Oversold (<30)" if hist['RSI'].iloc[-1] < 30 else "Neutral"))
    print(colored(f"MACD: {hist['MACD'].iloc[-1]:.2f}", "cyan"))
    print(colored(f"Stochastic %K: {hist['STOCH_%K'].iloc[-1]:.2f}", "cyan"))
    print(colored(f"Stochastic %D: {hist['STOCH_%D'].iloc[-1]:.2f}", "cyan"))
    print(colored(
        f"Bollinger Bands (Upper/Middle/Lower): {hist['Upper_BB'].iloc[-1]:.2f} / {hist['Middle_BB'].iloc[-1]:.2f} / {hist['Lower_BB'].iloc[-1]:.2f}",
        "cyan"))
    print(colored(f"Volatility (Std Dev): {hist['Std_Dev'].iloc[-1]:.2f}", "cyan"))
    print(colored(f"Volatility (ATR): {hist['ATR'].iloc[-1]:.2f}", "cyan"))
    print(colored(f"Historical Volatility (Annualized): {hist['Hist_Volatility'].iloc[-1]:.2f}", "cyan"))
    print(colored(f"Alpha (vs {benchmark_ticker}): {alpha:.4f}", "cyan"))
    print(colored(f"Beta (vs {benchmark_ticker}): {beta:.4f}", "cyan"))
    print(colored(f"Sharpe Ratio: {sharpe_ratio:.4f}", "cyan"))

    return hist


def price_visualization(hist):
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=hist, x=hist.index, y='Close', label='Price', color='royalblue')
    sns.lineplot(data=hist, x=hist.index, y='SMA_20', label='20D SMA', color='orange', linestyle='--')
    sns.lineplot(data=hist, x=hist.index, y='EMA_20', label='20D EMA', color='green', linestyle='--')
    plt.title('Price & Moving Averages', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    cursor = mplcursors.cursor(hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(
        f"Date: {hist.index[sel.target.index].strftime('%Y-%m-%d')}\n"
        f"Price: {hist['Close'].iloc[sel.target.index]:.2f}\n"
        f"20D SMA: {hist['SMA_20'].iloc[sel.target.index]:.2f}\n"
        f"20D EMA: {hist['EMA_20'].iloc[sel.target.index]:.2f}"
    ))

    plt.tight_layout()
    plt.show()


def technical_visualization(hist):
    fig, ax = plt.subplots(5, 1, figsize=(14, 18))

    # Price and Moving Averages
    sns.lineplot(data=hist, x=hist.index, y='Close', label='Price', color='royalblue', ax=ax[0])
    sns.lineplot(data=hist, x=hist.index, y='SMA_20', label='20D SMA', color='orange', linestyle='--', ax=ax[0])
    sns.lineplot(data=hist, x=hist.index, y='EMA_20', label='20D EMA', color='green', linestyle='--', ax=ax[0])
    ax[0].set_title('Price & Moving Averages')

    # RSI
    sns.lineplot(data=hist, x=hist.index, y='RSI', label='RSI', color='purple', ax=ax[1])
    ax[1].axhline(70, linestyle='--', color='red')
    ax[1].axhline(30, linestyle='--', color='green')
    ax[1].set_title('Relative Strength Index (14)')

    # MACD
    sns.lineplot(data=hist, x=hist.index, y='MACD', label='MACD', color='blue', ax=ax[2])
    sns.lineplot(data=hist, x=hist.index, y='MACD_Signal', label='Signal', color='orange', ax=ax[2])
    ax[2].bar(hist.index, hist['MACD_Hist'], label='Histogram',
              color=np.where(hist['MACD_Hist'] < 0, 'red', 'green'))
    ax[2].set_title('MACD (12, 26, 9)')

    # Bollinger Bands
    sns.lineplot(data=hist, x=hist.index, y='Close', label='Price', color='royalblue', ax=ax[3])
    sns.lineplot(data=hist, x=hist.index, y='Upper_BB', label='Upper BB', linestyle='--', color='red', ax=ax[3])
    sns.lineplot(data=hist, x=hist.index, y='Middle_BB', label='Middle BB', linestyle='--', color='green', ax=ax[3])
    sns.lineplot(data=hist, x=hist.index, y='Lower_BB', label='Lower BB', linestyle='--', color='red', ax=ax[3])
    ax[3].set_title('Bollinger Bands')

    # Stochastic Oscillator
    sns.lineplot(data=hist, x=hist.index, y='STOCH_%K', label='%K', color='blue', ax=ax[4])
    sns.lineplot(data=hist, x=hist.index, y='STOCH_%D', label='%D', color='orange', ax=ax[4])
    ax[4].axhline(80, linestyle='--', color='red')
    ax[4].axhline(20, linestyle='--', color='green')
    ax[4].set_title('Stochastic Oscillator')

    for a in ax:
        cursor = mplcursors.cursor(a, hover=True)
        cursor.connect("add", lambda sel: sel.annotation.set_text(
            f"Date: {hist.index[sel.target.index].strftime('%Y-%m-%d')}\n"
            f"Value: {sel.target[1]:.2f}"
        ))

    plt.tight_layout()
    plt.show()


def main():
    while True:
        asset, hist = get_ticker_data()
        display_basic_info(asset, hist)
        hist = calculate_indicators(hist)

        while True:
            print(colored("\n💡 COMMANDS: [PV] Price Visualization | [TV] Technical Visualization | [BACK] New Asset | [EXIT]", "magenta"))
            cmd = input(colored("Enter command: ", "yellow")).upper()

            if cmd == "PV":
                price_visualization(hist)
            elif cmd == "TV":
                technical_visualization(hist)
            elif cmd == "BACK":
                break
            elif cmd == "EXIT":
                print(colored("\n🚀 Thanks for using Aurora Terminal! Goodbye!\n", "cyan"))
                sys.exit()
            else:
                print(colored("Invalid command. Use PV/TV/BACK/EXIT", "red"))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(colored("\n🚀 Session terminated. Goodbye!", "cyan"))
        sys.exit()
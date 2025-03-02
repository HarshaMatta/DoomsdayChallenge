import pandas as pd
import math
from scipy.stats import norm
import matplotlib.pyplot as plt

fileName = 'SMP_Max.csv'
strikeRatio = 1.05


def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        raise ValueError("S, K, T, and sigma must be greater than 0.")

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type.lower() == 'call':
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == 'put':
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    return price

def preprocess_data():
    """ Reads CSV and cleans data. """
    df = pd.read_csv(fileName)
    df = df.sort_index(ascending=True)# Sort by index in ascending order
    reversed_df = df.iloc[::-1]
    df = reversed_df.reset_index(drop=True)# Reset index


    # Convert date to datetime format
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")

    # Remove $ sign and convert to float
    for col in ["Close/Last", "Open", "High", "Low"]:
        df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)

    return df

def volatility(index, df):
    # Use a trailing window of the past 252 days (ensure index >= 252)
    if index < 252:
        raise ValueError("Index too low for trailing volatility calculation")
    prices = df['Close/Last'][index-252:index]
    log_returns = (prices / prices.shift(1)).apply(math.log).dropna()
    return log_returns.std() * math.sqrt(252)

def diffCheck(start_index):
    totalProfit = 0
    totalCost = 0
    netReturn = 0
    """ Runs Black-Scholes calculation and prints results. """
    df = preprocess_data()
    row_count = len(df)
    callPrices = []

    for i in range(start_index, row_count - 252):
        S = df['Close/Last'][i]
        K = S * strikeRatio  # Use a slightly different strike price (e.g., 5% higher)
        vol = volatility(i, df)
        callPrice = black_scholes_price(S, K, 1, (strikeRatio -1), vol, 'call')
        yearlyProfit = df['Close/Last'][i+252] - (strikeRatio * S)

        
        print(f"Index: {i}")
        print(f"Stock Price (S): {S}")
        print(f"Strike Price (K): {K}")
        print(f"Volatility: {vol}")
        print(f"Call Price: {callPrice}")
        print(f"Yearly Profit: {yearlyProfit}")
        print("-" * 30)
        
        callPrices.append(callPrice)
        print(df['Date'][i], "  ", callPrice,"   " ,yearlyProfit)
        netReturn += (max(df['Close/Last'][i+252] - (1.05 * S), 0))
        totalCost += callPrice
    print("-" * 30)
    print("-" * 30)

    print(f"Total Profit: {netReturn - totalCost}")
    print(f"Total Cost: {totalCost}")
    print(f"Return on investment: {(netReturn - totalCost) / totalCost}")
    years = ((row_count - 252) - start_index) / 252
    print(f"Years: {years}")
    print(f"Annualized returns: {(netReturn / totalCost)**(1/years)}")
    return totalProfit


def plot_call_price_vs_profit():
    f"""Plots the call price against the yearly profit for {fileName} options."""
    df = preprocess_data()
    row_count = len(df)
    call_prices = []
    yearly_profits = []

    # Loop through valid indices where a 252-day forward profit can be computed
    for i in range(252, row_count - 252):
        S = df['Close/Last'][i]
        K = S * strikeRatio
        vol = volatility(i, df)
        call_price = black_scholes_price(S, K, 1, (strikeRatio -1), vol, 'call')
        profit = df['Close/Last'][i+252] - (S * strikeRatio)
        
        call_prices.append(call_price)
        yearly_profits.append(profit)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(call_prices, yearly_profits, alpha=0.7)
    plt.xlabel("Call Price")
    plt.ylabel("Yearly Profit")
    plt.title(f"Call Price vs. Yearly Profit for {fileName} Options")
    plt.grid(True)
    return plt


def plot_time_series():
    """Plots call price and yearly profit as a function of time on the same graph."""
    df = preprocess_data()
    row_count = len(df)
    dates = []
    call_prices = []
    yearly_profits = []

    # Iterate over indices ensuring there's a 252-day window ahead
    for i in range(252, row_count - 252):
        date = df['Date'][i]
        S = df['Close/Last'][i]
        K = S * strikeRatio  # Strike price 5% higher than S
        vol = volatility(i, df)
        call_price = black_scholes_price(S, K, 1, (strikeRatio -1), vol, 'call')
        profit = df['Close/Last'][i+252] - (S * strikeRatio)

        dates.append(date)
        call_prices.append(call_price)
        yearly_profits.append(profit)

    plt.figure(figsize=(12, 6))
    plt.plot(dates, call_prices, label='Call Price', color='blue', linewidth=2)
    plt.plot(dates, yearly_profits, label='Yearly Profit', color='green', linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title(f"{fileName} Call Price and Yearly Profit Over Time")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

if __name__ == '__main__':
    

    print(diffCheck(252))
    
    
    plot_time_series()
    plot_call_price_vs_profit()
    
    #fig, ax = plt.subplots(2, 1, figsize=(10, 12))
    #ax[0] = plot_call_price_vs_profit() 
    #ax[1] = plot_time_series()
    plt.show()

import math
from scipy.stats import norm

#!/usr/bin/env python3
"""
This program prices European options using the Black-Scholes formula.
"""


def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the Black-Scholes price for a European call or put option.

    Parameters:
        S (float): Underlying asset price.
        K (float): Strike price.
        T (float): Time to maturity in years.
        r (float): Annual risk-free interest rate (e.g., 0.05 for 5%).
        sigma (float): Annual volatility of the underlying asset (e.g., 0.2 for 20%).
        option_type (str): 'call' or 'put'.

    Returns:
        float: Option price.
    """
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

def main():
    print("European Option Pricing using the Black-Scholes Formula\n")
    try:
        S = float(input("Enter underlying asset price (S): "))
        K = float(input("Enter strike price (K): "))
        T = float(input("Enter time to maturity in years (T): "))
        r = float(input("Enter risk-free interest rate (r) (e.g., 0.05 for 5%): "))
        sigma = float(input("Enter volatility (sigma) (e.g., 0.2 for 20%): "))
        option_type = input("Enter option type (call/put): ").strip().lower()

        price = black_scholes_price(S, K, T, r, sigma, option_type)
        print(f"\nThe {option_type} option price is: {price:.4f}")
    except Exception as e:
        print(f"Error: {e}")
        
        


if __name__ == '__main__':
    main()
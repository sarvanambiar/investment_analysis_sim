import os
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm

# Data Download and Preprocessing Functions
def download_data(stocks, market_index, start_date, end_date):
    data = yf.download(stocks + [market_index], start=start_date, end=end_date, interval='1mo')['Adj Close']
    return data

def load_and_clean_factors(mom_file, ff_file):
    mom_factors = pd.read_csv(mom_file, index_col=None)
    mom_factors['Date'] = pd.to_datetime(mom_factors['Date'], format='%Y%m').dt.date

    ff_factors = pd.read_csv(ff_file, index_col=None)
    ff_factors['Date'] = pd.to_datetime(ff_factors['Date'], format='%Y%m').dt.date

    factors_merged = pd.merge(ff_factors, mom_factors, on='Date', how='inner')
    factors_merged['RF'] = ((factors_merged['RF'] / 100) / 12)
    factors_merged['RF'] = np.log(factors_merged['RF'] + 1).dropna()
    return factors_merged

def calculate_log_returns(data):
    data.index = pd.to_datetime(data.index).date
    log_returns = np.log(data / data.shift(1)).dropna()
    return log_returns

def merge_data(log_returns, factors_merged):
    log_returns['Date'] = log_returns.index
    combined_data = pd.merge(log_returns, factors_merged, on='Date', how='inner')
    combined_data.set_index('Date', inplace=True)
    return combined_data

def calculate_risk_premiums(combined_data, stocks):
    stock_risk_premium_data = {}
    for stock in stocks:
        stock_risk_premium = combined_data[stock] - combined_data['RF']
        stock_risk_premium_data[stock] = stock_risk_premium

    stock_risk_premium_df = pd.DataFrame(stock_risk_premium_data)
    market_risk_premium = combined_data['SPY'] - combined_data['RF']
    return stock_risk_premium_df, market_risk_premium

def get_yahoo_betas(stocks):
    comparison = {}
    for stock in stocks:
        comparison[stock] = yf.Ticker(stock).info.get('beta', None)
    return comparison

def capm_analysis(stock_risk_premium_df, market_risk_premium, stocks):
    results = []
    for stock in stocks:
        stock_risk_premium = stock_risk_premium_df[stock]
        X = sm.add_constant(market_risk_premium)
        model = sm.OLS(stock_risk_premium, X).fit()
        alpha, beta = model.params
        alpha_pval = model.pvalues['const']
        results.append({'Stock': stock, 'Alpha': alpha, 'Beta': beta, 'Alpha p-value': alpha_pval})

    yahoo_betas = get_yahoo_betas(stocks)
    capm_results = pd.DataFrame(results)
    capm_results['Yahoo Beta'] = capm_results['Stock'].map(yahoo_betas)
    return capm_results

def fama_french_analysis(stock_risk_premium_df, combined_data, market_risk_premium, stocks):
    results_ff = []
    for stock in stocks:
        stock_risk_premium = stock_risk_premium_df[stock]
        X = sm.add_constant(pd.concat([market_risk_premium, combined_data[['SMB', 'HML', 'Mom']]], axis=1))
        model = sm.OLS(stock_risk_premium, X).fit()
        alpha = model.params['const']
        beta_mkt, beta_smb, beta_hml, beta_mom = model.params[1:]
        alpha_pval = model.pvalues['const']
        residual_ss = model.ssr
        residual_df = model.df_resid
        residual_var = residual_ss / residual_df
        results_ff.append({
            'Stock': stock, 'Alpha': alpha, 'Beta MKT': beta_mkt, 'Beta SMB': beta_smb,
            'Beta HML': beta_hml, 'Beta MOM': beta_mom, 'Alpha p-value': alpha_pval,
            'SSR': residual_ss, 'DF Residual': residual_df, 'Residual Variance': residual_var
        })

    yahoo_betas = get_yahoo_betas(stocks)
    ff_results = pd.DataFrame(results_ff)
    ff_results['Yahoo Beta'] = ff_results['Stock'].map(yahoo_betas)
    return ff_results

def compute_tangent_portfolio_weights(diff_returns_rf, cov_matrix):
    """
    Computes the tangent portfolio weights.

    Parameters:
    - diff_returns_rf: Excess returns over the risk-free rate
    - cov_matrix: Variance-covariance matrix of stock returns

    Returns:
    - Tangent portfolio weights as a numpy array
    """
    excess_returns = diff_returns_rf.to_numpy()
    cov_matrix_inv = np.linalg.inv(cov_matrix)
    tangent_numerator = np.dot(cov_matrix_inv, excess_returns)
    tangent_denominator = np.dot(np.ones(len(excess_returns)), tangent_numerator)
    tangent_weights = tangent_numerator / tangent_denominator
    return tangent_weights

def compute_sharpe_ratio_tangent_portfolio(tangent_weights, diff_returns_rf, cov_matrix):
    """
    Computes the Sharpe ratio of the tangent portfolio.

    Parameters:
    - tangent_weights: Weights of the tangent portfolio
    - diff_returns_rf: Excess returns over the risk-free rate
    - cov_matrix: Variance-covariance matrix of stock returns

    Returns:
    - Sharpe ratio of the tangent portfolio
    """
    tangent_port_risk_premium = np.sum(tangent_weights * diff_returns_rf)
    portfolio_variance = np.dot(tangent_weights.T, np.dot(cov_matrix, tangent_weights))
    portfolio_std_dev = np.sqrt(portfolio_variance)
    sharpe_ratio_tangent_port = tangent_port_risk_premium / portfolio_std_dev
    return sharpe_ratio_tangent_port

def compute_sharpe_ratio_sim(ff_results, market_risk_premium, combined_data):
    """
    Computes the Sharpe ratio of the SIM portfolio.

    Parameters:
    - ff_results: Results from the Fama-French analysis
    - market_risk_premium: Market risk premium series
    - combined_data: Combined dataset including returns and factors

    Returns:
    - Sharpe ratio of the SIM portfolio
    """
    initial_position = ff_results['Alpha'] / ff_results['Residual Variance']
    scaled_initial_position = initial_position / initial_position.sum()
    weighted_avg_alpha = np.average(ff_results['Alpha'], weights=scaled_initial_position)
    weighted_residual_variance = np.average(ff_results['Residual Variance'], weights=scaled_initial_position**2)

    sharpe_ratio_market = np.mean(market_risk_premium) / np.std(combined_data['SPY'])
    information_ratio_squared = (weighted_avg_alpha ** 2) / weighted_residual_variance
    sharpe_ratio_sim = np.sqrt(sharpe_ratio_market ** 2 + information_ratio_squared)
    return sharpe_ratio_sim


def compute_adjusted_tangent_weights(initial_margin, tanget_stocks, diff_returns_rf, cov_matrix, investment_amount):
    """
    Adjust tangent weights for the given investment amount.
    """
    inverse_initial_margin = np.where(initial_margin != 0, 1 / initial_margin, 0)
    identity_matrix = np.eye(len(tanget_stocks))

    tanget_stocks = pd.Series(tanget_stocks)
    inverse_initial_margin = pd.Series(inverse_initial_margin.flatten())
    one_over_k_sign = inverse_initial_margin * np.sign(tanget_stocks)

    one_over_k_sign = one_over_k_sign.to_numpy()
    A_1 = identity_matrix * one_over_k_sign
    A_1[A_1 == -0.0] = 0.0
    tanget_stocks = -tanget_stocks.to_numpy().reshape(-1, 1)

    A = np.hstack((A_1, tanget_stocks))
    new_row = np.append(np.ones(10), [0])
    A = np.vstack((A, new_row))
    c = np.zeros((len(tanget_stocks) + 1, 1))
    c[-1, 0] = investment_amount

    Z = np.matmul(np.linalg.inv(A), c)
    stock_value = (Z[:len(tanget_stocks)] / initial_margin) * np.sign(-tanget_stocks)

    tangent_weights = stock_value / np.sum(stock_value)
    return tangent_weights.flatten()

# Main Execution
stocks = sorted(['AAPL', 'BRK-B', 'COST', 'FIX', 'JNJ', 'JPM', 'LMT', 'NVDA', 'TSLA', 'XOM'])
market_index = 'SPY'
start_date = '2019-09-01'
end_date = '2024-10-01'

data = download_data(stocks, market_index, start_date, end_date)
factors_merged = load_and_clean_factors('F-F_Momentum_Factor.CSV', 'F-F_Research_Data_5_Factors_2x3.CSV')
log_returns = calculate_log_returns(data)
log_returns_stock = log_returns.drop(columns=['SPY'])
combined_data = merge_data(log_returns, factors_merged)
stock_risk_premium_df, market_risk_premium = calculate_risk_premiums(combined_data, stocks)

capm_results = capm_analysis(stock_risk_premium_df, market_risk_premium, stocks)
ff_results = fama_french_analysis(stock_risk_premium_df, combined_data, market_risk_premium, stocks)

# SIM Portfolio Sharpe Ratio
sharpe_ratio_sim = compute_sharpe_ratio_sim(ff_results, market_risk_premium, combined_data)

# Tangent Portfolio
diff_returns_rf = (combined_data[stocks].mean() - combined_data['RF'].mean())
cov_matrix = log_returns_stock.cov().to_numpy()
tanget_stocks = compute_tangent_portfolio_weights(diff_returns_rf, cov_matrix)
sharpe_ratio_tangent = compute_sharpe_ratio_tangent_portfolio(tanget_stocks, diff_returns_rf.to_numpy(), cov_matrix)

# Investor wants to allocate $10,000 across stocks
# initial margin=50% for all assets
initial_margin = np.full((len(tanget_stocks), 1), 0.5)
tangent_weights = compute_adjusted_tangent_weights(initial_margin, tanget_stocks, diff_returns_rf, cov_matrix, 10000)
sharpe_ratio_tangent_1 = compute_sharpe_ratio_tangent_portfolio(tangent_weights, diff_returns_rf.to_numpy(), cov_matrix)

# Investor does not get a margin loan for long positions, but does need to get one for shorting (set it at 50%)
# Recompute the SR
initial_margin = np.where(tanget_stocks < 0, 0.5, 1).reshape(-1, 1)
tangent_weights = compute_adjusted_tangent_weights(initial_margin, tanget_stocks, diff_returns_rf, cov_matrix, 10000)
sharpe_ratio_tangent_2 = compute_sharpe_ratio_tangent_portfolio(tangent_weights, diff_returns_rf.to_numpy(), cov_matrix)

#  Investor sets the initial margin at 75% for any position, even though the requirement is only 50%. 
initial_margin = np.full((10, 1), 0.75)
tangent_weights = compute_adjusted_tangent_weights(initial_margin, tanget_stocks, diff_returns_rf, cov_matrix, 10000)
sharpe_ratio_tangent_3 = compute_sharpe_ratio_tangent_portfolio(tangent_weights, diff_returns_rf.to_numpy(), cov_matrix)

# Display Results
print("CAPM Results:\n", capm_results)
print("Fama-French Results:\n", ff_results)
print("Sharpe Ratio of SIM Portfolio:", sharpe_ratio_sim)
print("Tangent Portfolio Weights:\n", tangent_weights)
print("Sharpe Ratio of Tangent Portfolio:", sharpe_ratio_tangent)
print("Sharpe Ratio of Tangent Portfolio 1:", sharpe_ratio_tangent_1)
print("Sharpe Ratio of Tangent Portfolio 2:", sharpe_ratio_tangent_2)
print("Sharpe Ratio of Tangent Portfolio 3:", sharpe_ratio_tangent_3)


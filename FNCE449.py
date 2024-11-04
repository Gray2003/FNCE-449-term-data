import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# List of stock symbols
stock_array = ['APA', 'BKR', 'CHRD', 'CHX', 'EXE', 'FANG', 'GPRE', 'PTEN', 'VNOM', 'WFRD']

# Initialize an empty DataFrame to store all income statements
all_income_statements = pd.DataFrame()

# Loop through each stock symbol in the array
for stock in stock_array:
    file_path = f'dataset/FNCE-449-term-data-main/FNCE 449 term project data/{stock}/{stock} IS.xlsx'
    df_income_statement = pd.read_excel(file_path)

    # Set column names and drop unnecessary rows as per your previous steps
    df_income_statement.columns = df_income_statement.iloc[6]
    df_income_statement = df_income_statement.drop(df_income_statement.index[0:8])

    # Transpose the DataFrame and set new column names
    df_income_statement = df_income_statement.transpose()
    df_income_statement.columns = df_income_statement.iloc[0]
    df_income_statement = df_income_statement.drop(df_income_statement.index[0])

    # Add calculated columns for Revenue and Operating Expenses
    df_income_statement['Revenue'] = df_income_statement['Sales']
    df_income_statement['Operating Expenses'] = (
        df_income_statement['Cost of Goods Sold (COGS) incl. D&A'] +
        df_income_statement['SG&A Expense'] +
        df_income_statement['Other SG&A']
    )

    # Reset index and rename the 'Per End' column
    df_income_statement = df_income_statement.reset_index()

    df_income_statement = df_income_statement.rename(columns={6: 'Per End'})

    # Add a column to identify the stock
    df_income_statement['Stock'] = stock

    df_income_statement = df_income_statement[['Stock', 'Per End', 'Revenue', 'Operating Expenses', 'Net Income']]

    # Append the current income statement to the combined DataFrame
    all_income_statements = pd.concat([all_income_statements, df_income_statement])

# Display the combined DataFrame
all_income_statements[['Stock', 'Per End', 'Revenue', 'Operating Expenses', 'Net Income']]

# Initialize an empty DataFrame to store all income statements
all_balance_sheets = pd.DataFrame()

# Loop through each stock symbol in the array
for stock in stock_array:
    file_path = f'dataset/FNCE-449-term-data-main/FNCE 449 term project data/{stock}/{stock} BS.xlsx'
    df_Balance_Sheet = pd.read_excel(file_path)

    # Set column names and drop unnecessary rows as per your previous steps
    df_Balance_Sheet.columns = df_Balance_Sheet.iloc[6]
    df_Balance_Sheet = df_Balance_Sheet.drop(df_Balance_Sheet.index[0:8])

    # Transpose the DataFrame and set new column names
    df_Balance_Sheet = df_Balance_Sheet.transpose()
    df_Balance_Sheet.columns = df_Balance_Sheet.iloc[0]
    df_Balance_Sheet = df_Balance_Sheet.drop(df_Balance_Sheet.index[0])

    # Reset index and rename the 'Per End' column
    df_Balance_Sheet = df_Balance_Sheet.reset_index()

    df_Balance_Sheet = df_Balance_Sheet.rename(columns={6: 'Per End'})

    # Add a column to identify the stock
    df_Balance_Sheet['Stock'] = stock

    df_Balance_Sheet = df_Balance_Sheet[['Stock', 'Per End', 'Total Assets', 'Total Liabilities', "Total Shareholders' Equity", 'Book Value per Share', 'ST Debt & Curr. Portion LT Debt', "Long-Term Debt", 'Total Equity']]

    # Append the current income statement to the combined DataFrame
    all_balance_sheets = pd.concat([all_balance_sheets, df_Balance_Sheet])

# Loop through each stock symbol and create a separate DataFrame
for stock in stock_array:
    # Filter the all_balance_sheets DataFrame for the current stock
    df_stock_Balance_Sheet = all_balance_sheets[all_balance_sheets['Stock'] == stock].copy()

    # Dynamically create a DataFrame variable name
    globals()[f'df_{stock}_Balance_Sheet'] = df_stock_Balance_Sheet

# Initialize an empty DataFrame to store all income statements
all_cash_flows = pd.DataFrame()

# Loop through each stock symbol in the array
for stock in stock_array:
    file_path = f'dataset/FNCE-449-term-data-main/FNCE 449 term project data/{stock}/{stock} CF.xlsx'
    df_Cash_Flow = pd.read_excel(file_path)

    # Set column names and drop unnecessary rows as per your previous steps
    df_Cash_Flow.columns = df_Cash_Flow.iloc[6]
    df_Cash_Flow = df_Cash_Flow.drop(df_Cash_Flow.index[0:8])

    # Transpose the DataFrame and set new column names
    df_Cash_Flow = df_Cash_Flow.transpose()
    df_Cash_Flow.columns = df_Cash_Flow.iloc[0]
    df_Cash_Flow = df_Cash_Flow.drop(df_Cash_Flow.index[0])

    # Reset index and rename the 'Per End' column
    df_Cash_Flow = df_Cash_Flow.reset_index()

    df_Cash_Flow = df_Cash_Flow.rename(columns={6: 'Per End'})

    # Add a column to identify the stock
    df_Cash_Flow['Stock'] = stock

    df_Cash_Flow = df_Cash_Flow[['Stock', 'Per End', 'Net Operating Cash Flow', 'Net Investing Cash Flow']]

    # Append the current income statement to the combined DataFrame
    all_cash_flows = pd.concat([all_cash_flows, df_Cash_Flow])

# Display the combined DataFrame
all_cash_flows[['Stock', 'Per End', 'Net Operating Cash Flow', 'Net Investing Cash Flow']]

# List of stock symbols
stock_array = ['APA', 'BKR', 'CHRD', 'CHX', 'EXE', 'FANG', 'GPRE', 'PTEN', 'VNOM', 'WFRD']

# Initialize an empty DataFrame to store all income statements
all_earning_surprises = pd.DataFrame()

# Loop through each stock symbol in the array
for stock in stock_array:
    file_path = f'dataset/FNCE-449-term-data-main/FNCE 449 term project data/{stock}/{stock} Earnings surprise.xlsx'

    pd_APA_test = pd.read_excel(f'dataset/FNCE-449-term-data-main/FNCE 449 original/{stock}/{stock} Earnings surprise.xlsx')
    pd_APA_test

    df_Earning_Surprise = pd.read_excel(f'dataset/FNCE-449-term-data-main/FNCE 449 term project data/{stock}/{stock} Earnings surprise.xlsx')
    df_Earning_Surprise = df_Earning_Surprise.drop(df_Earning_Surprise.index[0:1])

    pd_APA_test['Price'] = df_Earning_Surprise['Price']
    pd_APA_test

    df_Earning_Surprise = pd_APA_test

    df_Earning_Surprise = df_Earning_Surprise.drop(df_Earning_Surprise.index[0])

    df_Earning_Surprise['Per End'] = df_Earning_Surprise['Per End'].apply(lambda x: datetime.datetime.strptime(str(x), '%m/%y').strftime("%b '%y").upper())
    df_Earning_Surprise

    # Drop rows in column "Price" that has value NaN
    df_Earning_Surprise = df_Earning_Surprise.dropna(subset=['Price'])

    df_Earning_Surprise[['Per End','Reported','Price']]
    df_Earning_Surprise['P/E'] = df_Earning_Surprise['Price'] / df_Earning_Surprise['Reported']

    df_Earning_Surprise = pd.merge(df_Earning_Surprise, globals()[f'df_{stock}_Balance_Sheet'], on='Per End')

    for i in df_Earning_Surprise['Book Value per Share']:
        if i == 0:
            df_Earning_Surprise = df_Earning_Surprise.drop(df_Earning_Surprise[df_Earning_Surprise['Book Value per Share'] == i].index)

    df_Earning_Surprise['P/B'] = df_Earning_Surprise['Price'] / df_Earning_Surprise['Book Value per Share']
    df_Earning_Surprise

    # Projected earning is (Estimate[i] /Estimate [i+1])-1
    projected_earnings = (df_Earning_Surprise['Reported'].shift(1) / df_Earning_Surprise['Reported']) - 1
    projected_earnings
    df_Earning_Surprise['Projected Earning'] = projected_earnings
    df_Earning_Surprise

    df_Earning_Surprise['PEG'] = df_Earning_Surprise['P/E'] / df_Earning_Surprise['Projected Earning']

    df_Earning_Surprise[['Per End','Projected Earning','PEG']]

    df_Earning_Surprise['D/E'] = (df_Earning_Surprise["ST Debt & Curr. Portion LT Debt"] + df_Earning_Surprise["Long-Term Debt"]) / df_Earning_Surprise['Total Equity']

    # Add a column to identify the stock
    df_Earning_Surprise['Stock'] = stock

    df_Earning_Surprise = df_Earning_Surprise[['Stock', 'Per End', 'P/E', 'P/B', 'D/E', 'Reported', 'Estimate']]
    # Append the current income statement to the combined DataFrame
    all_earning_surprises = pd.concat([all_earning_surprises, df_Earning_Surprise])

# Display the combined DataFrame
df_Earning_Surprise_core = pd.merge(all_earning_surprises, all_balance_sheets, on=['Per End', 'Stock'])
df_Earning_Surprise_core = pd.merge(df_Earning_Surprise_core, all_income_statements, on=['Per End', 'Stock'])
df_Earning_Surprise_core = pd.merge(df_Earning_Surprise_core, all_cash_flows, on=['Per End', 'Stock'])
CL_Brent_price = pd.read_excel('dataset/FNCE-449-term-data-main/FNCE 449 original/CL Brent price.xlsx')
CL_Brent_price.rename(columns = {'Security':'Date','CO1 Comdty':'CL Brent Price','Unnamed: 2':'Quantity CL Brent'}, inplace = True)
CL_Brent_price = CL_Brent_price[6:]
CL_Brent_price['Date'] = pd.to_datetime(CL_Brent_price['Date'])
CL_Brent_price['Date'] = CL_Brent_price['Date'].dt.strftime("%b '%y").str.upper()
CL_Brent_price = CL_Brent_price[["Date","CL Brent Price"]]

CL_price = pd.read_excel('dataset/FNCE-449-term-data-main/FNCE 449 original/CL price.xlsx')
CL_price.rename(columns = {'Security':'Date','CL1 COMB Comdty':'CL WTI Price','Unnamed: 2':'Quantity CL'}, inplace = True)
CL_price = CL_price[6:]
CL_price['Date'] = pd.to_datetime(CL_price['Date'])
CL_price['Date'] = CL_price['Date'].dt.strftime("%b '%y").str.upper()
CL_price = CL_price[["Date","CL WTI Price"]]

NG_price = pd.read_excel('dataset/FNCE-449-term-data-main/FNCE 449 original/NG price.xlsx')
NG_price.rename(columns = {'Security':'Date','NG1 COMB Comdty':'NG Price','Unnamed: 2':'Quantity NG'}, inplace = True)
NG_price = NG_price[6:]
NG_price['Date'] = pd.to_datetime(NG_price['Date'])
NG_price['Date'] = NG_price['Date'].dt.strftime("%b '%y").str.upper()
NG_price = NG_price[["Date", "NG Price"]]

Fed_Fund_Rate = pd.read_excel('dataset/FNCE-449-term-data-main/FNCE 449 original/Federal Fund Rate.xlsx')
Fed_Fund_Rate.rename(columns = {'Security':'Date','FEDL01 Index':'Federal Fund Rate','Unnamed: 2':'Change'}, inplace = True)
Fed_Fund_Rate = Fed_Fund_Rate[5:]
Fed_Fund_Rate['Date'] = pd.to_datetime(Fed_Fund_Rate['Date'])
Fed_Fund_Rate['Date'] = Fed_Fund_Rate['Date'].dt.strftime("%b '%y").str.upper()
Fed_Fund_Rate = Fed_Fund_Rate[["Date", "Federal Fund Rate"]]

GDP_QoQ = pd.read_excel('dataset/FNCE-449-term-data-main/FNCE 449 original/GDP QoQ.xlsx')
GDP_QoQ.rename(columns = {'Security':'Date','GDP CQOQ Index':'GDP growth','Unnamed: 2':'ACTUAL_RELEASE_gdp_growth'}, inplace = True)
GDP_QoQ = GDP_QoQ[5:]
GDP_QoQ['Date'] = pd.to_datetime(GDP_QoQ['Date'])
GDP_QoQ['Date'] = GDP_QoQ['Date'].dt.strftime("%b '%y").str.upper()
GDP_QoQ = GDP_QoQ[["Date", "ACTUAL_RELEASE_gdp_growth"]]

Inflation_CPI = pd.read_excel('dataset/FNCE-449-term-data-main/FNCE 449 original/Inflation CPI.xlsx')
Inflation_CPI.rename(columns = {'Security':'Date','CPI CHNG Index':'CPI_growth','Unnamed: 2':'Change'}, inplace = True)
Inflation_CPI = Inflation_CPI[5:]
Inflation_CPI['Date'] = pd.to_datetime(Inflation_CPI['Date'])
Inflation_CPI['Date'] = Inflation_CPI['Date'].dt.strftime("%b '%y").str.upper()
Inflation_CPI = Inflation_CPI[["Date", "CPI_growth"]]

Inflation_PCE = pd.read_excel('dataset/FNCE-449-term-data-main/FNCE 449 original/Inflation PCE.xlsx')
Inflation_PCE.rename(columns = {'Security':'Date','PCE CMOM Index':'PCE_growth','Unnamed: 2':'ACTUAL_RELEASE'}, inplace = True)
Inflation_PCE = Inflation_PCE[5:]
Inflation_PCE['Date'] = pd.to_datetime(Inflation_PCE['Date'])
Inflation_PCE['Date'] = Inflation_PCE['Date'].dt.strftime("%b '%y").str.upper()
Inflation_PCE = Inflation_PCE[["Date", "PCE_growth"]]

US_SA_NFP = pd.read_excel('dataset/FNCE-449-term-data-main/FNCE 449 original/US SA NFP.xlsx')
US_SA_NFP.rename(columns = {'Security':'Date','NFP T Index':'NFP'}, inplace = True)
US_SA_NFP = US_SA_NFP[5:]
US_SA_NFP['Date'] = pd.to_datetime(US_SA_NFP['Date'])
US_SA_NFP['Date'] = US_SA_NFP['Date'].dt.strftime("%b '%y").str.upper()
US_SA_NFP = US_SA_NFP[["Date", "NFP"]]

regression_input = pd.merge(CL_Brent_price, CL_price, on='Date')
regression_input = pd.merge(regression_input, Fed_Fund_Rate, on='Date')
regression_input = pd.merge(regression_input, GDP_QoQ, on='Date')
regression_input = pd.merge(regression_input, Inflation_CPI, on='Date')
regression_input = pd.merge(regression_input, Inflation_PCE, on='Date')
regression_input = pd.merge(regression_input, NG_price, on='Date')
regression_input = pd.merge(regression_input, US_SA_NFP, on='Date')

regression_input = pd.merge(regression_input, df_Earning_Surprise_core, left_on='Date',right_on='Per End')
regression_input.to_csv('regression_input.csv', index=False)

# Load the data
data = pd.read_csv('regression_input.csv')
data = data.iloc[::-1]

# Replace infinite values with NaN
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Now drop rows with NaN values
data.dropna(inplace=True)

# Convert stock names to numbers
stock_array = ['APA', 'BKR', 'CHRD', 'CHX', 'EXE', 'FANG', 'GPRE', 'PTEN', 'VNOM', 'WFRD']
stock_to_num = {stock: idx for idx, stock in enumerate(stock_array)}
data['Stock_Num'] = data['Stock'].map(stock_to_num)

# Drop 'Date' as it should not be used as a feature for training
# However, keep the sorted order for time-based splitting
data = data.reset_index(drop=True)
dates = data['Date']

# Define features (X) and target variable (y)
X = data.drop(columns=['Date', 'Reported', 'Per End', 'Stock'])
y = data['Reported']

# Split data into time-based train and test sets (80% train, 20% test)
split_index = int(len(data) * 0.9035)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
dates_test = dates.iloc[split_index:]

    # 2. With Normalization
# Normalize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the random forest model with normalized data
rf_scaling = RandomForestRegressor(random_state=42)
rf_scaling.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_scaling = rf_scaling.predict(X_test_scaled)

# Calculate RMSE
rmse_scaling = np.sqrt(mean_squared_error(y_test, y_pred_scaling))
print(f'RMSE with normalization: {rmse_scaling}')

# Print the dates, true values, and predictions for the test set
print("\nPredictions with normalization:")
for date, ticker_num, true_val, analyst_val, pred_val in zip(dates_test, X_test['Stock_Num'], y_test, X_test['Estimate'], y_pred_scaling):
    print(f"Date: {date}, Ticker: {stock_array[ticker_num]}, True Value: {round(true_val,2)}, Analyst Value: {round(analyst_val,2)}, Predicted Value: {round(pred_val,2)}")

date = ['Sep 23', 'Dec 23', 'Mar 24']

for i in range(10):
    # Extract the subset of data for each range
    dates = dates_test.iloc[[i, i+10, i+20]]
    true_values = y_test.iloc[[i, i+10, i+20]]
    analyst_values = X_test['Estimate'].iloc[[i, i+10, i+20]]
    predicted_values = y_pred_scaling[[i, i+10, i+20]]  # Accessing NumPy array directly
    tickers = [stock_array[num] for num in X_test['Stock_Num'].iloc[[i, i+10, i+20]]]
    
    # Set up the figure
    x = np.arange(len(dates))  # Position on the x-axis for each stock

    plt.figure(figsize=(12, 6))

    # Plot each set of values as dots
    plt.scatter(x, true_values, color='blue', label='True Value', s=100, marker='o', facecolors='none')
    plt.scatter(x, analyst_values, color='orange', label='Analyst Estimate', s=100, marker='o', facecolors='none')
    plt.scatter(x, predicted_values, color='green', label='Predicted Value', s=100, marker='o', facecolors='none')

    # Add labels, title, and legend
    plt.xlabel('Stocks')
    plt.ylabel('Values')
    plt.title(f'Predictions and True Values for Stocks in {stock_array[9-i]}')
    plt.xticks(x, date, rotation=45, ha='right')  # Label each x position with the stock ticker
    plt.legend()
    plt.tight_layout()
    plt.show()
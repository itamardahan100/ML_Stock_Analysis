import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

tickers = ['^GSPC', '^IXIC']
results = []

for ticker in tickers:
    print(f"\n--- Analyzing {ticker} ---")
    
    # 1. Data Fetching
    data = yf.download(ticker, start='2015-01-01', end='2026-07-28')
    data['Returns'] = data['Close'].pct_change()
    data['Target'] = (data['Returns'] > 0).astype(int)
    data.dropna(inplace=True)

    # 2. Visualization (Saving the plot)
    plt.figure() 
    data['Close'].plot(title=f'{ticker} Historical Price', figsize=(10, 6))
    
   
    filename = f"{ticker.replace('^', '')}_plot.png" 
    plt.savefig(filename)
    plt.close() 
    
    print(f"Visualization saved as {filename}")

    # 3. Model Training and Comparison
    X = data[['Returns']].shift(1).dropna()
    y = data['Target'].iloc[1:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    models = {'Logistic Regression': LogisticRegression(), 'SVM (RBF)': SVC(kernel='rbf')}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        results.append({'Ticker': ticker, 'Model': name, 'Accuracy': acc})

# 4. Results Table
df_results = pd.DataFrame(results)
print("\n--- Model Comparison Table ---")
print(df_results)

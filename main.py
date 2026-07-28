import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Fetch historical data for S&P 500
data = yf.download('^GSPC', start='2015-01-01', end='2026-07-28')

# Feature engineering: daily returns and binary target
data['Returns'] = data['Close'].pct_change()
data['Target'] = (data['Returns'] > 0).astype(int)
data.dropna(inplace=True)

# Train test split for linear predictor
X = data[['Returns']].shift(1).dropna()
y = data['Target'].iloc[1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Model training and evaluation
model = LogisticRegression()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model accuracy is: {accuracy:.4f}")

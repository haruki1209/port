import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 取得する銘柄のティッカー
tickers = ['SPY', 'AAPL', 'MSFT', 'GOOGL']

# 期間設定
start_date = '2018-01-01'
end_date = '2021-01-01'

# yfinanceを使って株価データを取得
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# 各銘柄の1日リターン（株価の変化率）を計算
returns = data.pct_change().dropna()  # pct_change()で1日リターンを計算

# 異なるポートフォリオの重みをランダムに生成
num_portfolios = 10000  # ポートフォリオの数
results = np.zeros((3, num_portfolios))  # リターン、リスク、シャープレシオを保存する配列

for i in range(num_portfolios):
    # 重みをランダムに生成（合計が1になるように）
    weights = np.random.random(4)
    weights /= np.sum(weights)

    # ポートフォリオのリターンとリスクを計算
    portfolio_return = np.sum(weights * returns.mean()) * 252  # 年間リターン
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))  # 年間リスク

    # 結果を保存
    results[0,i] = portfolio_return
    results[1,i] = portfolio_risk
    results[2,i] = portfolio_return / portfolio_risk  # シャープレシオ

# 結果をデータフレームに変換
results_df = pd.DataFrame(results.T, columns=['Return', 'Risk', 'Sharpe Ratio'])

# 散布図を作成
plt.figure(figsize=(10, 6))
scatter = plt.scatter(results_df.Risk, results_df.Return, c=results_df['Sharpe Ratio'], cmap='viridis', marker='o')

# 凡例を追加
plt.title('risk-return', fontsize=14)
plt.xlabel('standard deviation', fontsize=12)
plt.ylabel('return', fontsize=12)
plt.colorbar(scatter, label='Sharpe ratio')  # シャープレシオを色で表示

# 各銘柄の色を表示するための凡例を作成
for i, ticker in enumerate(tickers):
    plt.scatter([], [], c=scatter.cmap(scatter.norm(results_df['Sharpe Ratio'].iloc[i])), label=ticker)

# 凡例を右上に配置
plt.legend(title='Brand name', loc='upper left', bbox_to_anchor=(1.2, 1))  # 右上に配置

# レイアウトを調整
plt.subplots_adjust(right=0.9)  # 右側に間隔を追加

plt.show()

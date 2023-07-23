# pylint: disable-all
# Mean Reversion Strategy Simulator using Pairs Trading of 2 securities
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import sys


class MeanReversionSimulator:

    def __init__(self):
        self.securities = {}
        self.time_frame = 30
        self.entry_threshold = 0.5
        self.exit_threshold = 0.5
        self.asset1 = "None"
        self.asset2 = "None"

    def main_menu(self):
        self.print_menu()
        option = input("=>")
        if option == "1":
            # Code for Securities input
            self.get_securities()
            self.main_menu()
        elif option == "2":
            # Code for changing Time Frame
            self.time_frame = int(input("Enter Time Frame: "))
            self.main_menu()
        elif option == "3":
            # Code for changing Entry and Exit Threshold
            self.entry_threshold = float(input("Enter Entry Threshold: "))
            self.exit_threshold = float(input("Exit Entry Threshold: "))
            self.main_menu()
        elif option == "4":
            # Code for Simulated Trading (Backtesting using historical data)
            print("\nResult:")
            correlation = self.calculate_correlation(self.securities[self.asset1], self.securities[self.asset2])
            print(f"({self.asset1.upper()}:{self.asset2.upper()}) Correlation: {round(correlation, 3)}")
            profit = self.backtest()
            print(f"{self.asset1} to {self.asset2} Overall profit: {profit:.2f}%\n")
            self.main_menu()
        elif option == "5":
            # Code for Chart Visualization
            self.plot_bollinger_bands()
            self.main_menu()
        elif option == "6":
            # Code for exiting the system
            print("Exiting the application...")
            sys.exit()
        else:
            # Code for default case
            print("INVALID INPUT.  TRY AGAIN.")
            self.main_menu()

    def get_securities(self):
        self.asset1 = input("Enter 1st asset: ").upper()
        self.asset2 = input("Enter 2nd asset: ").upper()
        # sample asset ["BTC-USD", "ETH-USD", "AAPL", "GOOGL"]
        self.securities[self.asset1] = self.create_series(self.asset1, "10y")
        self.securities[self.asset2] = self.create_series(self.asset2, "10y")

    def print_menu(self):
        print(f"1) Securities: {self.asset1}  {self.asset2}")
        print(f"2) Time Frame: {self.time_frame}")
        print(f"3) Threshold: {self.entry_threshold}  {self.exit_threshold}")
        print(f"4) Simulate Trading")
        print("5) Charts")
        print("6) Main Menu")
        print("7) Exit")

    @staticmethod
    def create_series(ticker_input, period):
        ticker = yf.Ticker(ticker_input)
        data = ticker.history(period=period)
        return pd.Series(data["Close"])

    def backtest(self):
        signals = self.entry_exit_conditions(self.securities[self.asset1], self.securities[self.asset2],
                                             self.entry_threshold, self.exit_threshold, self.time_frame)
        daily_returns = self.calculate_daily_returns(self.securities[self.asset1], self.securities[self.asset2],
                                                     signals)
        cumulative_returns = (1 + daily_returns).cumprod()
        cumulative_returns = cumulative_returns.drop(cumulative_returns.index[-1])
        overall_profit = (cumulative_returns.iloc[-1] - 1) * 100
        return overall_profit

    @staticmethod
    def calculate_correlation(asset1, asset2):
        return asset2.corr(asset1)

    @staticmethod
    def entry_exit_conditions(asset1, asset2, entry_threshold, exit_threshold, time_frame):
        # Calculate Bollinger Bands
        window = time_frame  # Moving Average Time Frame
        asset1_sma = asset1.rolling(window=window, min_periods=1).mean()
        asset1_std = asset1.rolling(window=window, min_periods=1).std()
        asset2_sma = asset2.rolling(window=window, min_periods=1).mean()
        asset2_std = asset2.rolling(window=window, min_periods=1).std()

        # Generate trading signals
        signals = pd.Series(0, index=asset1.index)

        # Long asset1 and short asset2 when asset1 price is below the lower Bollinger Band and asset2 price is above the upper Bollinger Band
        signals[(asset1 < asset1_sma - entry_threshold * asset1_std) & (
                asset2 > asset2_sma + entry_threshold * asset2_std)] = 1

        # Short asset1 and long asset2 when asset1 price is above the upper Bollinger Band and asset2 price is below the lower Bollinger Band
        signals[(asset1 > asset1_sma + entry_threshold * asset1_std) & (
                asset2 < asset2_sma - entry_threshold * asset2_std)] = -1

        # Exit signals when asset1 and asset2 prices move back within the exit_threshold range of their respective means
        exit_condition = (asset1 >= asset1_sma - exit_threshold * asset1_std) & (
                asset1 <= asset1_sma + exit_threshold * asset1_std) & (
                                 asset2 >= asset2_sma - exit_threshold * asset2_std) & (
                                 asset2 <= asset2_sma + exit_threshold * asset2_std)
        signals[exit_condition] = 0

        return signals

    @staticmethod
    def calculate_daily_returns(asset1, asset2, signals):
        # Ensure all variables are Pandas Series or DataFrames
        signals = pd.Series(signals, index=asset1.index)  # Convert signals to a Pandas Series
        asset1_returns = asset1.pct_change().shift(-1)
        asset2_returns = asset2.pct_change().shift(-1)
        daily_returns = signals * (asset1_returns - asset2_returns)
        total_trades = (signals != 0).sum()
        print("Total Number of Trades:", total_trades)

        return daily_returns

    def plot_bollinger_bands(self):
        asset1_sma = self.securities[self.asset1].rolling(window=self.time_frame, min_periods=1).mean()
        asset1_std = self.securities[self.asset1].rolling(window=self.time_frame, min_periods=1).std()
        asset2_sma = self.securities[self.asset2].rolling(window=self.time_frame, min_periods=1).mean()
        asset2_std = self.securities[self.asset2].rolling(window=self.time_frame, min_periods=1).std()

        plt.figure(figsize=(12, 6))
        plt.plot(self.securities[self.asset1].index, self.securities[self.asset1], label=self.asset1, color="blue")
        plt.plot(self.securities[self.asset2].index, self.securities[self.asset2], label=self.asset2, color="green")
        plt.plot(self.securities[self.asset1].index, asset1_sma, label="SMA " + self.asset1, color="orange")
        plt.plot(self.securities[self.asset2].index, asset2_sma, label="SMA " + self.asset2, color="red")
        plt.fill_between(self.securities[self.asset1].index, asset1_sma - 2 * asset1_std, asset1_sma + 2 * asset1_std,
                         alpha=0.2, color="gray")
        plt.fill_between(self.securities[self.asset2].index, asset2_sma - 2 * asset2_std, asset2_sma + 2 * asset2_std,
                         alpha=0.2, color="gray")
        plt.fill_between(self.securities[self.asset1].index, asset1_sma - 2 * asset1_std, asset1_sma + 2 * asset1_std,
                         alpha=0.2, color="gray", label="Bollinger Bands " + self.asset1)
        plt.fill_between(self.securities[self.asset2].index, asset2_sma - 2 * asset2_std, asset2_sma + 2 * asset2_std,
                         alpha=0.2, color="gray", label="Bollinger Bands " + self.asset2)
        plt.title("Bollinger Bands")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    simulator = MeanReversionSimulator()
    simulator.main_menu()

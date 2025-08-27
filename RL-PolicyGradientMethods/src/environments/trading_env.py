"""
Trading Environment for Policy Gradient Methods
"""
import numpy as np
import pandas as pd
import yfinance as yf
from gymnasium import Env, spaces
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt


class TradingEnvironment(Env):
    """
    A trading environment for testing policy gradient methods
    
    The agent can take actions:
    0: Hold
    1: Buy
    2: Sell
    
    State includes:
    - Price features (OHLCV)
    - Technical indicators
    - Portfolio state
    """
    
    def __init__(self, symbol: str = "AAPL", start_date: str = "2020-01-01", 
                 end_date: str = "2023-01-01", initial_balance: float = 10000.0,
                 transaction_cost: float = 0.001, lookback_window: int = 20):
        """
        Initialize trading environment
        
        Args:
            symbol: Stock symbol to trade
            start_date: Start date for data
            end_date: End date for data
            initial_balance: Initial portfolio balance
            transaction_cost: Transaction cost as fraction
            lookback_window: Number of historical prices to include in state
        """
        super().__init__()
        
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        
        # Load data
        self.data = self._load_data(symbol, start_date, end_date)
        self.prices = self.data['Close'].values
        self.volumes = self.data['Volume'].values
        
        # Calculate technical indicators
        self._calculate_indicators()
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell
        
        # State includes: price features, technical indicators, portfolio state
        state_dim = (lookback_window * 5) + 10 + 3  # OHLCV + indicators + portfolio
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.max_steps = len(self.prices) - lookback_window - 1
        
        # Portfolio state
        self.balance = initial_balance
        self.shares = 0
        self.portfolio_value = initial_balance
        
        # Performance tracking
        self.trades = []
        self.portfolio_history = []
        
    def _load_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load stock data from Yahoo Finance"""
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            # Return dummy data if download fails
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            np.random.seed(42)
            prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.02)
            return pd.DataFrame({
                'Open': prices * (1 + np.random.randn(len(dates)) * 0.01),
                'High': prices * (1 + np.abs(np.random.randn(len(dates))) * 0.02),
                'Low': prices * (1 - np.abs(np.random.randn(len(dates))) * 0.02),
                'Close': prices,
                'Volume': np.random.randint(1000000, 10000000, len(dates))
            }, index=dates)
    
    def _calculate_indicators(self):
        """Calculate technical indicators"""
        # Simple Moving Averages
        self.data['SMA_5'] = self.data['Close'].rolling(window=5).mean()
        self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
        
        # RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        self.data['BB_middle'] = self.data['Close'].rolling(window=20).mean()
        bb_std = self.data['Close'].rolling(window=20).std()
        self.data['BB_upper'] = self.data['BB_middle'] + (bb_std * 2)
        self.data['BB_lower'] = self.data['BB_middle'] - (bb_std * 2)
        
        # MACD
        exp1 = self.data['Close'].ewm(span=12).mean()
        exp2 = self.data['Close'].ewm(span=26).mean()
        self.data['MACD'] = exp1 - exp2
        self.data['MACD_signal'] = self.data['MACD'].ewm(span=9).mean()
        
        # Fill NaN values
        self.data = self.data.fillna(method='bfill').fillna(method='ffill')
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        if self.current_step < self.lookback_window:
            # Pad with zeros if not enough history
            price_features = np.zeros((self.lookback_window, 5))
            indicators = np.zeros(10)
        else:
            # Get price features (OHLCV) for lookback window
            start_idx = self.current_step - self.lookback_window
            end_idx = self.current_step
            
            price_features = np.array([
                self.data['Open'].iloc[start_idx:end_idx].values,
                self.data['High'].iloc[start_idx:end_idx].values,
                self.data['Low'].iloc[start_idx:end_idx].values,
                self.data['Close'].iloc[start_idx:end_idx].values,
                self.data['Volume'].iloc[start_idx:end_idx].values
            ]).T
            
            # Normalize price features
            price_features = (price_features - price_features.mean(axis=0)) / (price_features.std(axis=0) + 1e-8)
            
            # Get current technical indicators
            current_idx = self.current_step
            indicators = np.array([
                self.data['SMA_5'].iloc[current_idx],
                self.data['SMA_20'].iloc[current_idx],
                self.data['RSI'].iloc[current_idx],
                self.data['BB_upper'].iloc[current_idx],
                self.data['BB_lower'].iloc[current_idx],
                self.data['BB_middle'].iloc[current_idx],
                self.data['MACD'].iloc[current_idx],
                self.data['MACD_signal'].iloc[current_idx],
                self.data['Volume'].iloc[current_idx],
                self.data['Close'].iloc[current_idx]
            ])
            
            # Normalize indicators
            indicators = (indicators - indicators.mean()) / (indicators.std() + 1e-8)
        
        # Portfolio state
        portfolio_state = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.shares / 100,  # Normalized shares
            self.portfolio_value / self.initial_balance  # Normalized portfolio value
        ])
        
        # Combine all features
        state = np.concatenate([
            price_features.flatten(),
            indicators,
            portfolio_state
        ])
        
        return state.astype(np.float32)
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        current_price = self.prices[self.current_step]
        return self.balance + (self.shares * current_price)
    
    def _execute_trade(self, action: int) -> float:
        """Execute trade and return reward"""
        current_price = self.prices[self.current_step]
        reward = 0
        
        if action == 1:  # Buy
            if self.balance > current_price:
                # Calculate number of shares to buy
                shares_to_buy = int(self.balance / (current_price * (1 + self.transaction_cost)))
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                    self.balance -= cost
                    self.shares += shares_to_buy
                    
                    # Record trade
                    self.trades.append({
                        'step': self.current_step,
                        'action': 'buy',
                        'price': current_price,
                        'shares': shares_to_buy,
                        'cost': cost
                    })
        
        elif action == 2:  # Sell
            if self.shares > 0:
                # Sell all shares
                proceeds = self.shares * current_price * (1 - self.transaction_cost)
                self.balance += proceeds
                
                # Record trade
                self.trades.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'price': current_price,
                    'shares': self.shares,
                    'proceeds': proceeds
                })
                
                self.shares = 0
        
        # Calculate reward based on portfolio value change
        new_portfolio_value = self._calculate_portfolio_value()
        reward = (new_portfolio_value - self.portfolio_value) / self.initial_balance
        self.portfolio_value = new_portfolio_value
        
        return reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        # Execute trade
        reward = self._execute_trade(action)
        
        # Update step
        self.current_step += 1
        
        # Get next state
        next_state = self._get_state()
        
        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Store portfolio value
        self.portfolio_history.append(self.portfolio_value)
        
        # Info
        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'shares': self.shares,
            'current_price': self.prices[self.current_step - 1] if self.current_step > 0 else 0
        }
        
        return next_state, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Reset episode tracking
        self.current_step = 0
        
        # Reset portfolio
        self.balance = self.initial_balance
        self.shares = 0
        self.portfolio_value = self.initial_balance
        
        # Reset performance tracking
        self.trades = []
        self.portfolio_history = [self.portfolio_value]
        
        # Get initial state
        state = self._get_state()
        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'shares': self.shares
        }
        
        return state, info
    
    def render(self, mode: str = 'human'):
        """Render the environment"""
        if mode == 'human':
            print(f"Step: {self.current_step}, Portfolio Value: {self.portfolio_value:.2f}, "
                  f"Balance: {self.balance:.2f}, Shares: {self.shares}")
    
    def plot_performance(self):
        """Plot trading performance"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot portfolio value over time
        ax1.plot(self.portfolio_history)
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True)
        
        # Plot stock price with buy/sell points
        ax2.plot(self.prices[:len(self.portfolio_history)], label='Stock Price', alpha=0.7)
        
        # Mark buy and sell points
        for trade in self.trades:
            if trade['action'] == 'buy':
                ax2.scatter(trade['step'], trade['price'], color='green', marker='^', s=100, label='Buy' if trade == self.trades[0] or trade['action'] != self.trades[self.trades.index(trade)-1]['action'] else "")
            else:
                ax2.scatter(trade['step'], trade['price'], color='red', marker='v', s=100, label='Sell' if trade == self.trades[0] or trade['action'] != self.trades[self.trades.index(trade)-1]['action'] else "")
        
        ax2.set_title('Stock Price with Trading Actions')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Price ($)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        if not self.portfolio_history:
            return {}
        
        # Total return
        total_return = (self.portfolio_history[-1] - self.initial_balance) / self.initial_balance
        
        # Sharpe ratio (simplified)
        returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)  # Annualized
        
        # Maximum drawdown
        peak = np.maximum.accumulate(self.portfolio_history)
        drawdown = (self.portfolio_history - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Win rate
        winning_trades = sum(1 for trade in self.trades if trade['action'] == 'sell' and 
                           trade['proceeds'] > trade['shares'] * self.prices[trade['step']])
        total_trades = len([t for t in self.trades if t['action'] == 'sell'])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.trades)
        }

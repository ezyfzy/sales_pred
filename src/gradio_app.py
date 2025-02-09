import gradio as gr
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from model_training import prepare_training_data, train_xgboost_model, predict_test_data
import yfinance as yf
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import zipfile
from sklearn.ensemble import RandomForestRegressor

class MLPipeline:
    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.best_model = None
        self.feature_importance = None
        self.study = None

    def upload_data(self, train_file, test_file):
        """Upload and save the data files"""
        os.makedirs('data', exist_ok=True)
        
        self.train_df = pd.read_csv(train_file.name)
        self.test_df = pd.read_csv(test_file.name)
        
        self.train_df.to_csv('data/prepared_train.csv', index=False)
        self.test_df.to_csv('data/prepared_test.csv', index=False)
        
        return f"Data uploaded successfully!\nTrain shape: {self.train_df.shape}\nTest shape: {self.test_df.shape}"

    def clean_and_impute(self):
        """Perform data cleaning and imputation"""
        if self.train_df is None or self.test_df is None:
            return "Please upload data first!"
        
        # Add your cleaning and imputation logic here
        # For example:
        self.train_df = self.train_df.fillna(self.train_df.mean(numeric_only=True))
        self.test_df = self.test_df.fillna(self.test_df.mean(numeric_only=True))
        
        return "Data cleaning and imputation completed!"

    def perform_eda(self):
        """Perform Exploratory Data Analysis"""
        if self.train_df is None:
            return "Please upload and clean data first!", None, None, None
        
        plots = []
        
        # 1. Numerical distributions
        numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns
        fig_dist = px.box(self.train_df, y=numeric_cols[:5], title="Distribution of Numerical Features")
        plots.append(fig_dist)
        
        # 2. Correlation heatmap
        corr = self.train_df.select_dtypes(include=[np.number]).corr()
        fig_corr = px.imshow(corr, title="Correlation Heatmap")
        plots.append(fig_corr)
        
        # 3. Missing values
        missing = self.train_df.isnull().sum()
        fig_missing = px.bar(x=missing.index, y=missing.values, title="Missing Values")
        plots.append(fig_missing)
        
        return "EDA completed!", plots[0], plots[1], plots[2]

    def engineer_features(self):
        """Perform feature engineering"""
        if self.train_df is None or self.test_df is None:
            return "Please complete previous steps first!"
        
        # Add your feature engineering logic here
        self.X_train, self.X_val, self.y_train, self.y_val = prepare_training_data(self.train_df)
        
        return f"Feature engineering completed!\nTraining features shape: {self.X_train.shape}"

    def tune_hyperparameters(self):
        """Perform hyperparameter tuning"""
        if self.X_train is None:
            return "Please complete feature engineering first!", None
        
        self.best_model, self.feature_importance, self.study = train_xgboost_model(
            self.X_train, self.X_val, self.y_train, self.y_val, n_trials=50
        )
        
        # Create optimization history plot
        history = pd.DataFrame({
            'Trial': range(len(self.study.trials)),
            'RMSE': [t.value for t in self.study.trials]
        })
        fig_history = px.line(
            history,
            x='Trial',
            y='RMSE',
            title='Optimization History'
        )
        
        return "Hyperparameter tuning completed!", fig_history

    def train_and_predict(self):
        """Train final model and make predictions"""
        if self.best_model is None:
            return "Please complete hyperparameter tuning first!", None, None, None, "", None
        
        # Make predictions
        predict_test_data(self.best_model, self.test_df)
        
        # Create visualizations
        # 1. Feature Importance Plot
        fig_importance = px.bar(
            self.feature_importance.head(10),
            x='importance',
            y='feature',
            orientation='h',
            title='Top 10 Important Features'
        )
        
        # 2. Actual vs Predicted Plot
        val_predictions = self.best_model.predict(self.X_val)
        fig_scatter = px.scatter(
            x=self.y_val,
            y=val_predictions,
            labels={'x': 'Actual Values', 'y': 'Predicted Values'},
            title='Actual vs Predicted Values (Validation Set)'
        )
        fig_scatter.add_trace(
            go.Scatter(
                x=[self.y_val.min(), self.y_val.max()],
                y=[self.y_val.min(), self.y_val.max()],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            )
        )
        
        # 3. Prediction Distribution Plot
        predictions_df = pd.read_csv('data/predictions.csv')
        fig_dist = px.histogram(
            predictions_df,
            x='Item_Outlet_Sales',
            title='Distribution of Predicted Sales'
        )
        
        # Get metrics
        train_pred = self.best_model.predict(self.X_train)
        val_pred = self.best_model.predict(self.X_val)
        
        train_rmse = np.sqrt(((train_pred - self.y_train) ** 2).mean())
        val_rmse = np.sqrt(((val_pred - self.y_val) ** 2).mean())
        
        metrics_text = f"""
        Model Performance Metrics:
        Training RMSE: {train_rmse:.2f}
        Validation RMSE: {val_rmse:.2f}
        
        Best Hyperparameters:
        """
        for key, value in self.study.best_params.items():
            metrics_text += f"\n{key}: {value}"
        
        return "Model training and prediction completed!", fig_importance, fig_scatter, fig_dist, metrics_text, "data/predictions.csv"

class StockAnalysisPipeline:
    def __init__(self):
        """Initialize the stock analysis pipeline"""
        self.nse_stocks = {
            'Reliance Industries': 'RELIANCE.NS',
            'TCS': 'TCS.NS',
            'HDFC Bank': 'HDFCBANK.NS',
            'Infosys': 'INFY.NS',
            'ICICI Bank': 'ICICIBANK.NS'
        }
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.stock_data = None
        self.selected_stock = None

    def prepare_technical_indicators(self, data):
        """Calculate all technical indicators"""
        try:
            df = data.copy()
            
            # Calculate Moving Averages
            df['MA5'] = df['Close'].rolling(window=5, min_periods=1).mean()
            df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
            df['MA50'] = df['Close'].rolling(window=50, min_periods=1).mean()
            df['MA200'] = df['Close'].rolling(window=200, min_periods=1).mean()
            
            # Calculate Bollinger Bands
            rolling_mean = df['Close'].rolling(window=20, min_periods=1).mean()
            rolling_std = df['Close'].rolling(window=20, min_periods=1).std()
            
            df['BB_middle'] = rolling_mean
            df['BB_upper'] = rolling_mean + (rolling_std * 2)
            df['BB_lower'] = rolling_mean - (rolling_std * 2)
            
            # Calculate Support and Resistance
            period = 20  # Look back period for support/resistance
            df['Support'] = df['Close'].rolling(window=period, min_periods=1).min()
            df['Resistance'] = df['Close'].rolling(window=period, min_periods=1).max()
            
            # Calculate RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 1e-9)  # Add small constant to avoid division by zero
            df['RSI'] = 100 - (100 / (1 + rs))
            df['RSI'] = df['RSI'].clip(0, 100)  # Clip RSI between 0 and 100
            
            # Calculate MACD
            exp1 = df['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
            
            # Calculate Fibonacci Retracement Levels
            max_price = df['High'].rolling(window=period, min_periods=1).max()
            min_price = df['Low'].rolling(window=period, min_periods=1).min()
            diff = max_price - min_price
            
            df['Fib_0.236'] = max_price - (diff * 0.236)
            df['Fib_0.382'] = max_price - (diff * 0.382)
            df['Fib_0.5'] = max_price - (diff * 0.5)
            df['Fib_0.618'] = max_price - (diff * 0.618)
            df['Fib_0.786'] = max_price - (diff * 0.786)
            
            # Fill NaN values
            df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
            
            return df
            
        except Exception as e:
            print(f"Error in prepare_technical_indicators: {str(e)}")
            raise

    def prepare_features(self, data):
        """Prepare features for model training"""
        try:
            # Create basic features
            features = pd.DataFrame()
            features['Close'] = data['Close']
            features['Volume'] = data['Volume']
            features['MA5'] = data['MA5']
            features['MA20'] = data['MA20']
            features['RSI'] = data['RSI']
            features['MACD'] = data['MACD']
            
            # Fill NaN values
            features = features.fillna(method='bfill').fillna(method='ffill').fillna(0)
            
            return features
            
        except Exception as e:
            print(f"Error in prepare_features: {str(e)}")
            print(f"Available columns: {data.columns.tolist()}")
            raise

    def train_model(self, X, y):
        """Train the prediction model"""
        try:
            if len(X) != len(y):
                raise ValueError("Feature and target arrays must have the same length")
                
            # Initialize model if not already done
            if self.model is None:
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
                
            # Train the model
            self.model.fit(X, y)
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            raise

    def predict_future(self, data, days=30):
        """Make future predictions"""
        try:
            # Create training features and target
            X = pd.DataFrame()
            X['Close'] = data['Close']
            X['Volume'] = data['Volume']
            X['MA5'] = data['MA5']
            X['MA20'] = data['MA20']
            X['RSI'] = data['RSI']
            X['MACD'] = data['MACD']
            
            # Prepare target (next day's price)
            y = data['Close'].shift(-1).dropna()
            X = X.iloc[:-1]  # Remove last row to match target length
            
            # Train model
            self.model.fit(X, y)
            
            # Prepare predictions
            predictions = []
            dates = []
            last_data = X.iloc[-1].copy()
            current_date = data.index[-1]
            
            # Make predictions
            for _ in range(days):
                current_date = current_date + pd.Timedelta(days=1)
                dates.append(current_date)
                
                # Make single prediction
                pred = self.model.predict(last_data.values.reshape(1, -1))[0]
                predictions.append(pred)
                
                # Update last_data for next prediction
                last_data['Close'] = pred
                last_data['MA5'] = (last_data['MA5'] * 4 + pred) / 5
                last_data['MA20'] = (last_data['MA20'] * 19 + pred) / 20
                # Keep other features relatively stable
                
            # Create prediction DataFrame
            future_df = pd.DataFrame({
                'Date': dates,
                'Predicted_Close': predictions
            })
            
            return future_df
            
        except Exception as e:
            print(f"Error in predict_future: {str(e)}")
            print(f"Data shape: {data.shape}")
            print(f"Available columns: {data.columns.tolist()}")
            raise

    def fetch_stock_data(self, stock_name, start_date, end_date):
        """Fetch stock data and calculate indicators"""
        if stock_name not in self.nse_stocks:
            return "Invalid stock selection!", None
        
        ticker = self.nse_stocks[stock_name]
        self.selected_stock = stock_name
        
        try:
            # Fetch data
            self.stock_data = yf.download(ticker, start=start_date, end=end_date)
            
            if self.stock_data.empty:
                return "No data available for the selected period!", None
            
            # Calculate technical indicators
            self.stock_data = self.prepare_technical_indicators(self.stock_data)
            
            # Verify data
            if self.stock_data is None:
                return "Error calculating technical indicators!", None
            
            return "Data fetched and processed successfully!", self.stock_data
            
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return f"Error fetching data: {str(e)}", None

    def download_data(self):
        """Download the stock data as CSV"""
        if self.stock_data is None:
            return None
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save to CSV
        csv_path = f'data/{self.selected_stock.replace(" ", "_")}_data.csv'
        self.stock_data.to_csv(csv_path)
        return csv_path

    def create_price_plot(self, future_pred_df):
        """Create price plot with predictions and technical indicators"""
        fig = go.Figure()
        
        # Historical prices
        fig.add_trace(go.Candlestick(
            x=self.stock_data.index,
            open=self.stock_data['Open'],
            high=self.stock_data['High'],
            low=self.stock_data['Low'],
            close=self.stock_data['Close'],
            name='Price'
        ))
        
        # 200-day Moving Average
        fig.add_trace(go.Scatter(
            x=self.stock_data.index,
            y=self.stock_data['MA200'],
            mode='lines',
            name='200-day MA',
            line=dict(color='blue', width=1)
        ))
        
        # Support and Resistance
        fig.add_trace(go.Scatter(
            x=self.stock_data.index,
            y=self.stock_data['Support'],
            mode='lines',
            name='Support',
            line=dict(color='green', width=1, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=self.stock_data.index,
            y=self.stock_data['Resistance'],
            mode='lines',
            name='Resistance',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        # Fibonacci Retracement Levels
        colors = ['rgba(255,0,0,0.3)', 'rgba(255,165,0,0.3)', 'rgba(255,255,0,0.3)', 
                  'rgba(0,255,0,0.3)', 'rgba(0,0,255,0.3)']
        levels = ['Fib_0.236', 'Fib_0.382', 'Fib_0.5', 'Fib_0.618', 'Fib_0.786']
        
        for level, color in zip(levels, colors):
            fig.add_trace(go.Scatter(
                x=self.stock_data.index,
                y=self.stock_data[level],
                mode='lines',
                name=f'Fib {level.split("_")[1]}',
                line=dict(color=color, width=1)
            ))
        
        # Future predictions
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(future_pred_df['Date']),
            y=future_pred_df['Predicted_Close'],
            mode='lines',
            name='Predicted',
            line=dict(color='purple', dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{self.selected_stock} Price Chart with Technical Indicators',
            yaxis_title='Price (₹)',
            xaxis_title='Date',
            template='plotly_white',
            hovermode='x unified',
            showlegend=True
        )
        
        return fig

    def create_volume_plot(self):
        """Create volume analysis plot"""
        fig = go.Figure()
        
        # Volume bars
        fig.add_trace(go.Bar(
            x=self.stock_data.index,
            y=self.stock_data['Volume'],
            name='Volume'
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{self.selected_stock} Trading Volume',
            yaxis_title='Volume',
            xaxis_title='Date',
            template='plotly_white',
            showlegend=True
        )
        
        return fig

    def create_prediction_plot(self, future_pred_df):
        """Create prediction trend plot"""
        fig = go.Figure()
        
        # Historical prices (last 30 days)
        fig.add_trace(go.Scatter(
            x=self.stock_data.index[-30:],
            y=self.stock_data['Close'][-30:],
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Future predictions
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(future_pred_df['Date']),
            y=future_pred_df['Predicted_Close'],
            mode='lines',
            name='Predicted',
            line=dict(color='red', dash='dash')
        ))
        
        # Add confidence intervals (optional)
        last_price = self.stock_data['Close'].iloc[-1]
        std_dev = self.stock_data['Close'].std()
        upper_bound = future_pred_df['Predicted_Close'] + std_dev
        lower_bound = future_pred_df['Predicted_Close'] - std_dev
        
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(future_pred_df['Date']),
            y=upper_bound,
            fill=None,
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(future_pred_df['Date']),
            y=lower_bound,
            fill='tonexty',
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),
            name='Confidence Interval',
            fillcolor='rgba(255,0,0,0.1)'
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{self.selected_stock} Price Prediction (Next 30 Days)',
            yaxis_title='Price (₹)',
            xaxis_title='Date',
            template='plotly_white',
            hovermode='x unified',
            showlegend=True
        )
        
        return fig

    def calculate_metrics(self, future_pred_df):
        """Calculate and format analysis metrics"""
        try:
            if self.stock_data is None or len(self.stock_data) == 0:
                return "Error: No historical data available"
            
            if future_pred_df is None or len(future_pred_df) == 0:
                return "Error: No predictions available"
            
            # Get current price and technical levels
            current_price = float(self.stock_data['Close'].iloc[-1])
            ma200 = float(self.stock_data['MA200'].iloc[-1])
            support = float(self.stock_data['Support'].iloc[-1])
            resistance = float(self.stock_data['Resistance'].iloc[-1])
            rsi = float(self.stock_data['RSI'].iloc[-1])
            macd = float(self.stock_data['MACD'].iloc[-1])
            signal = float(self.stock_data['Signal_Line'].iloc[-1])
            
            # Get trend status
            trend = "Bullish" if current_price > ma200 else "Bearish"
            
            # Format the metrics string
            metrics = f"""
            Analysis for {self.selected_stock}:
            
            Current Price: ₹{current_price:.2f}
            
            Technical Levels:
            Support: ₹{support:.2f}
            Resistance: ₹{resistance:.2f}
            200-day MA: ₹{ma200:.2f}
            
            Indicators:
            Trend: {trend}
            RSI: {rsi:.2f}
            MACD: {macd:.2f}
            Signal Line: {signal:.2f}
            
            Price Position:
            Distance from 200-day MA: {((current_price - ma200) / ma200 * 100):.2f}%
            Distance from Support: {((current_price - support) / support * 100):.2f}%
            Distance from Resistance: {((current_price - resistance) / resistance * 100):.2f}%
            
            Trading Signals:
            {self._get_trading_signals()}
            """
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return f"Error calculating metrics: {str(e)}"

    def _get_trading_signals(self):
        """Generate trading signals based on technical indicators"""
        try:
            signals = []
            
            current_price = float(self.stock_data['Close'].iloc[-1])
            ma200 = float(self.stock_data['MA200'].iloc[-1])
            support = float(self.stock_data['Support'].iloc[-1])
            resistance = float(self.stock_data['Resistance'].iloc[-1])
            rsi = float(self.stock_data['RSI'].iloc[-1])
            macd = float(self.stock_data['MACD'].iloc[-1])
            signal_line = float(self.stock_data['Signal_Line'].iloc[-1])
            
            # Trend Analysis
            if current_price > ma200:
                signals.append("TREND: Bullish (Price > 200 MA)")
            else:
                signals.append("TREND: Bearish (Price < 200 MA)")
            
            # Support/Resistance Analysis
            if current_price < support * 1.02:
                signals.append("SUPPORT: Price near support - Potential buying zone")
            elif current_price > resistance * 0.98:
                signals.append("RESISTANCE: Price near resistance - Potential selling zone")
            
            # RSI Analysis
            if rsi > 70:
                signals.append("RSI: Overbought condition (RSI > 70)")
            elif rsi < 30:
                signals.append("RSI: Oversold condition (RSI < 30)")
            
            # MACD Analysis
            if macd > signal_line:
                signals.append("MACD: Bullish signal (MACD > Signal Line)")
            else:
                signals.append("MACD: Bearish signal (MACD < Signal Line)")
            
            return "\n".join(signals)
            
        except Exception as e:
            return f"Error generating signals: {str(e)}"

    def fetch_and_predict(self, stock_name, start_date, end_date):
        """Fetch data and make predictions"""
        try:
            # Fetch and process data
            status, data = self.fetch_stock_data(stock_name, start_date, end_date)
            if "Error" in status or data is None:
                return status, None, None, None, None, None, None
            
            # Make predictions
            try:
                future_pred_df = self.predict_future(data)
                
                # Save files
                os.makedirs('data', exist_ok=True)
                
                raw_data_path = f'data/{stock_name.replace(" ", "_")}_raw_data.csv'
                self.stock_data.to_csv(raw_data_path)
                
                analyzed_data_path = f'data/{stock_name.replace(" ", "_")}_analyzed_data.csv'
                self.stock_data.to_csv(analyzed_data_path)
                
                predictions_path = f'data/{stock_name.replace(" ", "_")}_predictions.csv'
                future_pred_df.to_csv(predictions_path, index=False)
                
                # Create ZIP file
                zip_file, status_msg = self.create_and_download_zip(stock_name)
                
                return ("Analysis completed successfully!", 
                        self.create_technical_analysis_plots(future_pred_df),
                        self.create_volume_analysis(),
                        self.create_momentum_indicators(),
                        self.calculate_metrics(future_pred_df),
                        zip_file,
                        status_msg)
                
            except Exception as e:
                print(f"Error in prediction step: {str(e)}")
                return f"Error in prediction: {str(e)}", None, None, None, None, None, None
            
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            return f"Error during analysis: {str(e)}", None, None, None, None, None, None

    def create_technical_analysis_plots(self, future_pred_df):
        """Create comprehensive technical analysis plots with TradingView-style layout"""
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=3, cols=2,
            shared_xaxes=True,
            vertical_spacing=0.03,
            horizontal_spacing=0.03,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=(
                f'{self.selected_stock} Price Analysis', 'Volume Profile',
                'Technical Indicators', 'Price Momentum',
                'Market Depth', 'Trading Activity'
            )
        )

        # Main candlestick chart with enhanced styling
        fig.add_trace(
            go.Candlestick(
                x=self.stock_data.index,
                open=self.stock_data['Open'],
                high=self.stock_data['High'],
                low=self.stock_data['Low'],
                close=self.stock_data['Close'],
                name='OHLC',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )

        # Add Moving Averages with gradient colors
        ma_colors = {
            'MA5': 'rgba(255,255,0,0.7)',
            'MA20': 'rgba(0,255,0,0.7)',
            'MA50': 'rgba(255,165,0,0.7)',
            'MA200': 'rgba(255,0,0,0.7)'
        }

        for ma, color in ma_colors.items():
            fig.add_trace(
                go.Scatter(
                    x=self.stock_data.index,
                    y=self.stock_data[ma],
                    name=ma,
                    line=dict(color=color, width=1),
                    opacity=0.8
                ),
                row=1, col=1
            )

        # Add Bollinger Bands with fill
        fig.add_trace(
            go.Scatter(
                x=self.stock_data.index,
                y=self.stock_data['BB_upper'],
                name='BB Upper',
                line=dict(color='rgba(173,204,255,0.7)', dash='dash'),
                fill=None
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=self.stock_data.index,
                y=self.stock_data['BB_lower'],
                name='BB Lower',
                line=dict(color='rgba(173,204,255,0.7)', dash='dash'),
                fill='tonexty',
                fillcolor='rgba(173,204,255,0.1)'
            ),
            row=1, col=1
        )

        # Volume bars with color based on price movement
        colors = ['#26a69a' if close > open else '#ef5350'
                  for close, open in zip(self.stock_data['Close'], self.stock_data['Open'])]

        fig.add_trace(
            go.Bar(
                x=self.stock_data.index,
                y=self.stock_data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.8
            ),
            row=2, col=1
        )

        # RSI
        fig.add_trace(
            go.Scatter(
                x=self.stock_data.index,
                y=self.stock_data['RSI'],
                name='RSI',
                line=dict(color='#2962ff', width=1)
            ),
            row=2, col=2
        )

        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=2)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=2)

        # MACD
        fig.add_trace(
            go.Scatter(
                x=self.stock_data.index,
                y=self.stock_data['MACD'],
                name='MACD',
                line=dict(color='#2962ff', width=1)
            ),
            row=3, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=self.stock_data.index,
                y=self.stock_data['Signal_Line'],
                name='Signal Line',
                line=dict(color='#ff6d00', width=1)
            ),
            row=3, col=1
        )

        # Add future predictions
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(future_pred_df['Date']),
                y=future_pred_df['Predicted_Close'],
                name='Prediction',
                line=dict(color='purple', dash='dash', width=2),
                opacity=0.8
            ),
            row=1, col=1
        )

        # Update layout with TradingView-style design
        fig.update_layout(
            title_text=f"Technical Analysis for {self.selected_stock}",
            template='plotly_dark',
            plot_bgcolor='rgba(19,23,34,1)',
            paper_bgcolor='rgba(19,23,34,1)',
            font=dict(color='#e1e1e1'),
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            legend=dict(
                bgcolor='rgba(19,23,34,0.6)',
                bordercolor='rgba(255,255,255,0.1)',
                borderwidth=1
            ),
            margin=dict(l=50, r=50, t=85, b=50)
        )

        # Update axes
        fig.update_xaxes(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.1)',
            rangeslider_visible=False,
            showgrid=True
        )

        fig.update_yaxes(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.1)',
            showgrid=True
        )

        return fig

    def create_volume_analysis(self):
        """Create volume analysis chart"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Volume Analysis', 'Volume MA Comparison'),
            row_heights=[0.7, 0.3]
        )
        
        # Calculate volume moving averages
        self.stock_data['Volume_MA5'] = self.stock_data['Volume'].rolling(window=5).mean()
        self.stock_data['Volume_MA20'] = self.stock_data['Volume'].rolling(window=20).mean()
        
        # Main volume chart
        colors = ['green' if close > open else 'red' 
                  for close, open in zip(self.stock_data['Close'], self.stock_data['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=self.stock_data.index,
                y=self.stock_data['Volume'],
                name='Volume',
                marker_color=colors
            ),
            row=1, col=1
        )
        
        # Volume moving averages
        fig.add_trace(
            go.Scatter(
                x=self.stock_data.index,
                y=self.stock_data['Volume_MA5'],
                name='Volume MA5',
                line=dict(color='yellow')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.stock_data.index,
                y=self.stock_data['Volume_MA20'],
                name='Volume MA20',
                line=dict(color='orange')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title_text=f"Volume Analysis for {self.selected_stock}",
            height=600,
            showlegend=True,
            template='plotly_dark'
        )
        
        return fig

    def create_momentum_indicators(self):
        """Create momentum indicators chart"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('RSI', 'MACD'),
            row_heights=[0.5, 0.5]
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(
                x=self.stock_data.index,
                y=self.stock_data['RSI'],
                name='RSI',
                line=dict(color='cyan')
            ),
            row=1, col=1
        )
        
        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
        
        # MACD
        fig.add_trace(
            go.Scatter(
                x=self.stock_data.index,
                y=self.stock_data['MACD'],
                name='MACD',
                line=dict(color='blue')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.stock_data.index,
                y=self.stock_data['Signal_Line'],
                name='Signal Line',
                line=dict(color='orange')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title_text=f"Momentum Indicators for {self.selected_stock}",
            height=600,
            showlegend=True,
            template='plotly_dark'
        )
        
        return fig

    def create_and_download_zip(self, stock_name):
        """Create and download a zip file containing all data files"""
        try:
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            # Define file paths
            raw_data_path = f'data/{stock_name.replace(" ", "_")}_raw_data.csv'
            analyzed_data_path = f'data/{stock_name.replace(" ", "_")}_analyzed_data.csv'
            predictions_path = f'data/{stock_name.replace(" ", "_")}_predictions.csv'
            zip_path = f'data/{stock_name.replace(" ", "_")}_all_data.zip'
            
            # Check if files exist
            files_to_zip = []
            if os.path.exists(raw_data_path):
                files_to_zip.append(('Raw Data', raw_data_path))
            if os.path.exists(analyzed_data_path):
                files_to_zip.append(('Analyzed Data', analyzed_data_path))
            if os.path.exists(predictions_path):
                files_to_zip.append(('Predictions', predictions_path))
            
            if not files_to_zip:
                return None, "No data files found. Please run analysis first!"
            
            try:
                # Create ZIP file
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_label, file_path in files_to_zip:
                        file_name = os.path.basename(file_path)
                        zipf.write(file_path, file_name)
            
                # Return the file path first, then the status message
                if os.path.exists(zip_path):
                    return zip_path, f"ZIP file created successfully with {len(files_to_zip)} files!"
                else:
                    return None, "Error: ZIP file not created"
                
            except Exception as e:
                print(f"Error creating ZIP file: {str(e)}")
                return None, f"Error creating ZIP file: {str(e)}"
                
        except Exception as e:
            print(f"Error in create_and_download_zip: {str(e)}")
            return None, f"Error preparing download: {str(e)}"

    def create_gradio_interface(self):
        """Create Gradio interface"""
        with gr.Blocks() as app:
            gr.Markdown("# Stock Market Analysis and Prediction")
            
            # Input components
            with gr.Row():
                stock_dropdown = gr.Dropdown(
                    choices=list(self.nse_stocks.keys()),
                    label="Select Stock",
                    value=list(self.nse_stocks.keys())[0]
                )
            
            with gr.Row():
                start_date = gr.Textbox(
                    label="Start Date (YYYY-MM-DD)",
                    value=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                )
                end_date = gr.Textbox(
                    label="End Date (YYYY-MM-DD)",
                    value=datetime.now().strftime('%Y-%m-%d')
                )
            
            # Main action buttons
            with gr.Row():
                analyze_btn = gr.Button("Analyze Stock", variant="primary", size="lg")
            
            # Analysis section
            gr.Markdown("## Analysis Results")
            output_message = gr.Textbox(label="Analysis Status", interactive=False)
            
            with gr.Row():
                price_plot = gr.Plot(label="Price Analysis")
                volume_plot = gr.Plot(label="Volume Analysis")
            
            with gr.Row():
                momentum_plot = gr.Plot(label="Momentum Indicators")
            
            metrics_text = gr.Textbox(label="Analysis Metrics", interactive=False)
            
            # Download section
            gr.Markdown("## Download Data")
            with gr.Row():
                download_zip_btn = gr.Button("Download All Data as ZIP", variant="secondary")
            
            # Separate components for status and file
            download_status = gr.Textbox(label="Download Status", interactive=False)
            download_file = gr.File(label="Download ZIP File", interactive=True)
            
            # Button click events
            analyze_btn.click(
                fn=self.fetch_and_predict,
                inputs=[stock_dropdown, start_date, end_date],
                outputs=[output_message, price_plot, volume_plot, momentum_plot, 
                        metrics_text, download_status, download_file]
            )
            
            download_zip_btn.click(
                fn=self.create_and_download_zip,
                inputs=[stock_dropdown],
                outputs=[download_status, download_file]
            )
            
            return app

    def format_files_for_display(self, files_list):
        """Format files list for display in dataframe"""
        if not files_list:
            return []
        
        display_data = []
        for file in files_list:
            display_data.append([
                file['name'],
                file['size'],
                file['last_modified'],
                f"📥 [Download]({file['path']})"
            ])
        
        return display_data

    def run_app(self):
        """Run the Gradio app"""
        app = self.create_gradio_interface()
        app.launch(share=True)

# Create Gradio interface
with gr.Blocks(title="Stock Market Analysis") as app:
    gr.Markdown("# Indian Stock Market Analysis and Prediction")
    
    pipeline = StockAnalysisPipeline()
    
    with gr.Tab("Stock Analysis"):
        with gr.Row():
            with gr.Column():
                stock_dropdown = gr.Dropdown(
                    choices=list(pipeline.nse_stocks.keys()),
                    label="Select Stock",
                    value="Reliance Industries"
                )
                start_date = gr.Textbox(
                    label="Start Date (YYYY-MM-DD)",
                    value=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                )
                end_date = gr.Textbox(
                    label="End Date (YYYY-MM-DD)",
                    value=datetime.now().strftime('%Y-%m-%d')
                )
                analyze_btn = gr.Button("Analyze Stock")
        
        status_output = gr.Textbox(label="Status")
        
        with gr.Row():
            with gr.Column():
                price_plot = gr.Plot(label="Price Chart with Predictions")
                volume_plot = gr.Plot(label="Volume Analysis")
                prediction_plot = gr.Plot(label="Future Predictions")
        
        metrics_output = gr.Textbox(label="Analysis Metrics", lines=10)
        
        with gr.Row():
            download_btn = gr.Button("Download Predictions")
            predictions_output = gr.File(label="Download Predictions CSV")
    
    # Set up button clicks
    analyze_btn.click(
        fn=pipeline.fetch_and_predict,
        inputs=[stock_dropdown, start_date, end_date],
        outputs=[
            status_output,
            price_plot,
            volume_plot,
            prediction_plot,
            metrics_output,
            predictions_output
        ]
    )
    
    download_btn.click(
        fn=pipeline.download_data,
        inputs=[],
        outputs=[predictions_output]
    )

if __name__ == "__main__":
    app.launch(share=True) 
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
        # Dictionary of Indian stock symbols with their correct Yahoo Finance tickers
        self.nse_stocks = {
            'Reliance Industries': 'RELIANCE.NS',  # Fixed from 'RELIANCE'
            'TCS': 'TCS.NS',
            'HDFC Bank': 'HDFCBANK.NS',
            'Infosys': 'INFY.NS',
            'ICICI Bank': 'ICICIBANK.NS',
            'HUL': 'HINDUNILVR.NS',
            'ITC': 'ITC.NS',
            'SBI': 'SBIN.NS',
            'Bharti Airtel': 'BHARTIARTL.NS',
            'Axis Bank': 'AXISBANK.NS',
            # Adding more major NSE stocks
            'L&T': 'LT.NS',
            'Bajaj Finance': 'BAJFINANCE.NS',
            'Asian Paints': 'ASIANPAINT.NS',
            'HCL Tech': 'HCLTECH.NS',
            'Maruti Suzuki': 'MARUTI.NS',
            'Kotak Bank': 'KOTAKBANK.NS',
            'Wipro': 'WIPRO.NS',
            'Sun Pharma': 'SUNPHARMA.NS',
            'Power Grid': 'POWERGRID.NS',
            'NTPC': 'NTPC.NS'
        }
        self.stock_data = None
        self.selected_stock = None
        self.model = None
        self.scaler = None

    def prepare_features(self, df):
        """Prepare features for prediction"""
        df = df.copy()
        
        try:
            # Replace infinite values with NaN
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Technical indicators with error handling
            try:
                # Moving averages
                df['MA5'] = df['Close'].rolling(window=5, min_periods=1).mean()
                df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
                df['MA200'] = df['Close'].rolling(window=200, min_periods=1).mean()  # Added 200-day MA
                
                # Support and Resistance levels (using rolling min/max)
                df['Support'] = df['Low'].rolling(window=20, min_periods=1).min()
                df['Resistance'] = df['High'].rolling(window=20, min_periods=1).max()
                
                # Fibonacci Retracement Levels
                high = df['High'].rolling(window=20, min_periods=1).max()
                low = df['Low'].rolling(window=20, min_periods=1).min()
                diff = high - low
                df['Fib_0.236'] = high - (diff * 0.236)
                df['Fib_0.382'] = high - (diff * 0.382)
                df['Fib_0.5'] = high - (diff * 0.5)
                df['Fib_0.618'] = high - (diff * 0.618)
                df['Fib_0.786'] = high - (diff * 0.786)
                
                # RSI
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
                rs = gain / (loss + 1e-9)  # Add small constant to prevent division by zero
                df['RSI'] = 100 - (100 / (1 + rs))
                df['RSI'] = df['RSI'].clip(0, 100)  # Clip RSI between 0 and 100
                
                # MACD
                exp1 = df['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
                exp2 = df['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
                df['MACD'] = exp1 - exp2
                
                # Bollinger Bands
                df['20dSTD'] = df['Close'].rolling(window=20, min_periods=1).std()
                df['Upper_Band'] = df['MA20'] + (df['20dSTD'] * 2)
                df['Lower_Band'] = df['MA20'] - (df['20dSTD'] * 2)
                
                # Price changes (as percentages, clipped to prevent extreme values)
                df['Price_Change'] = df['Close'].pct_change().fillna(0).clip(-1, 1)
                df['Price_Change_5d'] = df['Close'].pct_change(periods=5).fillna(0).clip(-1, 1)
                
                # Volume features (normalized)
                mean_volume = df['Volume'].mean()
                std_volume = df['Volume'].std()
                df['Volume_Normalized'] = ((df['Volume'] - mean_volume) / (std_volume + 1e-9)).clip(-5, 5)
                df['Volume_Change'] = df['Volume'].pct_change().fillna(0).clip(-1, 1)
                df['Volume_MA5'] = df['Volume'].rolling(window=5, min_periods=1).mean()
                
            except Exception as e:
                print(f"Error calculating technical indicators: {str(e)}")
                raise
            
            # Fill NaN values with appropriate methods
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Select features for prediction
            feature_columns = [
                'Open', 'High', 'Low', 'Close', 'Volume_Normalized',
                'MA5', 'MA20', 'RSI', 'MACD',
                'Upper_Band', 'Lower_Band',
                'Price_Change', 'Price_Change_5d',
                'Volume_Change', 'Volume_MA5'
            ]
            
            # Normalize price-based features by dividing by the current price
            price_based_features = ['Open', 'High', 'Low', 'Close', 'MA5', 'MA20', 'Upper_Band', 'Lower_Band']
            for feature in price_based_features:
                df[feature] = df[feature] / df['Close'].mean()
            
            # Final check for infinite values
            result = df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Clip extreme values
            result = result.clip(-1e6, 1e6)
            
            return result
        
        except Exception as e:
            print(f"Error in prepare_features: {str(e)}")
            raise

    def train_model(self, X, y):
        """Train XGBoost model for prediction"""
        try:
            from sklearn.preprocessing import RobustScaler
            from sklearn.model_selection import train_test_split
            import xgboost as xgb
            
            # Use RobustScaler instead of StandardScaler to handle outliers better
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Check for any remaining infinite values
            if not np.all(np.isfinite(X_scaled)):
                raise ValueError("Infinite values found after scaling")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train model with more robust parameters
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=42,
                early_stopping_rounds=10
            )
            
            # Add error handling for training
            try:
                self.model.fit(
                    X_train, 
                    y_train,
                    eval_set=[(X_test, y_test)],
                    verbose=False
                )
            except Exception as e:
                print(f"Error during model training: {str(e)}")
                raise
            
        except Exception as e:
            print(f"Error in train_model: {str(e)}")
            raise

    def predict_future(self, last_data, days=30):
        """Predict future prices"""
        future_predictions = []
        current_data = last_data.copy()
        
        for _ in range(days):
            # Prepare features for prediction
            features = self.prepare_features(current_data)
            last_features = features.iloc[-1:]
            
            # Scale features
            scaled_features = self.scaler.transform(last_features)
            
            # Make prediction
            prediction = self.model.predict(scaled_features)[0]
            
            # Create new row with predicted close price
            new_row = current_data.iloc[-1:].copy()
            new_row.index = [new_row.index[-1] + pd.Timedelta(days=1)]
            new_row['Close'] = prediction
            new_row['Open'] = prediction
            new_row['High'] = prediction
            new_row['Low'] = prediction
            new_row['Volume'] = current_data['Volume'].mean()
            
            # Append prediction
            future_predictions.append({
                'Date': new_row.index[0],
                'Predicted_Close': prediction
            })
            
            # Update current data for next prediction
            current_data = pd.concat([current_data, new_row])
        
        return pd.DataFrame(future_predictions)

    def fetch_stock_data(self, stock_name, start_date, end_date):
        """Fetch stock data from Yahoo Finance"""
        if stock_name not in self.nse_stocks:
            return "Invalid stock selection!", None, None, None, None
        
        ticker = self.nse_stocks[stock_name]
        self.selected_stock = stock_name
        
        try:
            # Fetch data
            self.stock_data = yf.download(ticker, start=start_date, end=end_date)
            
            if self.stock_data.empty:
                return "No data available for the selected period!", None, None, None, None
            
            # Calculate technical indicators
            # Moving averages
            self.stock_data['MA200'] = self.stock_data['Close'].rolling(window=200, min_periods=1).mean()
            self.stock_data['MA20'] = self.stock_data['Close'].rolling(window=20, min_periods=1).mean()
            
            # Support and Resistance levels
            self.stock_data['Support'] = self.stock_data['Low'].rolling(window=20, min_periods=1).min()
            self.stock_data['Resistance'] = self.stock_data['High'].rolling(window=20, min_periods=1).max()
            
            # Fibonacci Retracement Levels
            high = self.stock_data['High'].rolling(window=20, min_periods=1).max()
            low = self.stock_data['Low'].rolling(window=20, min_periods=1).min()
            diff = high - low
            self.stock_data['Fib_0.236'] = high - (diff * 0.236)
            self.stock_data['Fib_0.382'] = high - (diff * 0.382)
            self.stock_data['Fib_0.5'] = high - (diff * 0.5)
            self.stock_data['Fib_0.618'] = high - (diff * 0.618)
            self.stock_data['Fib_0.786'] = high - (diff * 0.786)
            
            # Create price chart
            fig_price = go.Figure()
            fig_price.add_trace(go.Candlestick(
                x=self.stock_data.index,
                open=self.stock_data['Open'],
                high=self.stock_data['High'],
                low=self.stock_data['Low'],
                close=self.stock_data['Close'],
                name='Price'
            ))
            
            # Add 200-day MA
            fig_price.add_trace(go.Scatter(
                x=self.stock_data.index,
                y=self.stock_data['MA200'],
                mode='lines',
                name='200-day MA',
                line=dict(color='blue', width=1)
            ))
            
            # Add Support and Resistance
            fig_price.add_trace(go.Scatter(
                x=self.stock_data.index,
                y=self.stock_data['Support'],
                mode='lines',
                name='Support',
                line=dict(color='green', width=1, dash='dash')
            ))
            
            fig_price.add_trace(go.Scatter(
                x=self.stock_data.index,
                y=self.stock_data['Resistance'],
                mode='lines',
                name='Resistance',
                line=dict(color='red', width=1, dash='dash')
            ))
            
            # Add Fibonacci levels
            colors = ['rgba(255,0,0,0.3)', 'rgba(255,165,0,0.3)', 'rgba(255,255,0,0.3)', 
                      'rgba(0,255,0,0.3)', 'rgba(0,0,255,0.3)']
            levels = ['Fib_0.236', 'Fib_0.382', 'Fib_0.5', 'Fib_0.618', 'Fib_0.786']
            
            for level, color in zip(levels, colors):
                fig_price.add_trace(go.Scatter(
                    x=self.stock_data.index,
                    y=self.stock_data[level],
                    mode='lines',
                    name=f'Fib {level.split("_")[1]}',
                    line=dict(color=color, width=1)
                ))
            
            fig_price.update_layout(
                title=f'{stock_name} Price Chart with Technical Indicators',
                yaxis_title='Price (₹)',
                xaxis_title='Date',
                template='plotly_white',
                hovermode='x unified',
                showlegend=True
            )
            
            # Create volume chart
            fig_volume = px.bar(
                self.stock_data,
                x=self.stock_data.index,
                y='Volume',
                title=f'{stock_name} Volume'
            )
            
            # Calculate daily returns
            self.stock_data['Returns'] = self.stock_data['Close'].pct_change()
            fig_returns = px.histogram(
                self.stock_data,
                x='Returns',
                title='Distribution of Daily Returns'
            )
            
            # Calculate metrics
            current_price = float(self.stock_data['Close'].iloc[-1])
            ma200 = float(self.stock_data['MA200'].iloc[-1])
            support = float(self.stock_data['Support'].iloc[-1])
            resistance = float(self.stock_data['Resistance'].iloc[-1])
            
            metrics = f"""
            Analysis for {stock_name}:
            
            Current Price: ₹{current_price:.2f}
            200-day MA: ₹{ma200:.2f}
            Trend: {"Bullish" if current_price > ma200 else "Bearish"}
            
            Technical Levels:
            Support: ₹{support:.2f}
            Resistance: ₹{resistance:.2f}
            
            Distance from Levels:
            200-day MA: {((current_price - ma200) / ma200 * 100):.2f}%
            Support: {((current_price - support) / support * 100):.2f}%
            Resistance: {((current_price - resistance) / resistance * 100):.2f}%
            """
            
            return "Data fetched successfully!", fig_price, fig_volume, fig_returns, metrics
            
        except Exception as e:
            return f"Error fetching data: {str(e)}", None, None, None, None

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
            
            # Get Fibonacci levels
            fib_levels = {
                '23.6%': float(self.stock_data['Fib_0.236'].iloc[-1]),
                '38.2%': float(self.stock_data['Fib_0.382'].iloc[-1]),
                '50.0%': float(self.stock_data['Fib_0.5'].iloc[-1]),
                '61.8%': float(self.stock_data['Fib_0.618'].iloc[-1]),
                '78.6%': float(self.stock_data['Fib_0.786'].iloc[-1])
            }
            
            # Get trend status
            trend = "Bullish" if current_price > ma200 else "Bearish"
            
            # Format the metrics string
            metrics = f"""
            Analysis for {self.selected_stock}:
            
            Current Price: ₹{current_price:.2f}
            200-day MA: ₹{ma200:.2f}
            Trend: {trend}
            
            Technical Levels:
            Support: ₹{support:.2f}
            Resistance: ₹{resistance:.2f}
            
            Fibonacci Retracement Levels:
            23.6%: ₹{fib_levels['23.6%']:.2f}
            38.2%: ₹{fib_levels['38.2%']:.2f}
            50.0%: ₹{fib_levels['50.0%']:.2f}
            61.8%: ₹{fib_levels['61.8%']:.2f}
            78.6%: ₹{fib_levels['78.6%']:.2f}
            
            Price Position:
            Distance from 200-day MA: {((current_price - ma200) / ma200 * 100):.2f}%
            Distance from Support: {((current_price - support) / support * 100):.2f}%
            Distance from Resistance: {((current_price - resistance) / resistance * 100):.2f}%
            
            {self._get_trading_signals()}
            """
            
            return metrics
            
        except Exception as e:
            return f"Error calculating metrics: {str(e)}"

    def _get_trading_signals(self):
        """Generate trading signals based on technical indicators"""
        try:
            # Get latest values
            current_price = float(self.stock_data['Close'].iloc[-1])
            ma200 = float(self.stock_data['MA200'].iloc[-1])
            ma30 = float(self.stock_data['MA30'].iloc[-1])
            ma20 = float(self.stock_data['MA20'].iloc[-1])
            support = float(self.stock_data['Support'].iloc[-1])
            resistance = float(self.stock_data['Resistance'].iloc[-1])
            rsi = float(self.stock_data['RSI'].iloc[-1])
            macd = float(self.stock_data['MACD'].iloc[-1])
            signal_line = float(self.stock_data['Signal_Line'].iloc[-1])
            
            signals = []
            
            # Trend analysis
            if current_price > ma200:
                signals.append("LONG TERM TREND: Bullish (Price > 200 MA)")
            else:
                signals.append("LONG TERM TREND: Bearish (Price < 200 MA)")
            
            if current_price > ma30:
                signals.append("MEDIUM TERM TREND: Bullish (Price > 30 MA)")
            else:
                signals.append("MEDIUM TERM TREND: Bearish (Price < 30 MA)")
            
            # Support/Resistance analysis
            if current_price < support * 1.02:
                signals.append("SUPPORT: Price near support level - Potential buying zone")
            elif current_price > resistance * 0.98:
                signals.append("RESISTANCE: Price near resistance level - Potential selling zone")
            
            # Moving Average Crossovers
            ma5 = float(self.stock_data['MA5'].iloc[-1])
            ma5_prev = float(self.stock_data['MA5'].iloc[-2])
            ma20_prev = float(self.stock_data['MA20'].iloc[-2])
            
            if ma5 > ma20 and ma5_prev < ma20_prev:
                signals.append("CROSSOVER: Golden Cross (5 MA crossed above 20 MA) - Bullish signal")
            elif ma5 < ma20 and ma5_prev > ma20_prev:
                signals.append("CROSSOVER: Death Cross (5 MA crossed below 20 MA) - Bearish signal")
            
            # RSI Analysis
            if rsi > 70:
                signals.append("RSI: Overbought condition (RSI > 70) - Consider taking profits")
            elif rsi < 30:
                signals.append("RSI: Oversold condition (RSI < 30) - Consider buying")
            
            # MACD Analysis
            macd_prev = float(self.stock_data['MACD'].iloc[-2])
            signal_line_prev = float(self.stock_data['Signal_Line'].iloc[-2])
            
            if macd > signal_line and macd_prev < signal_line_prev:
                signals.append("MACD: Bullish crossover - Consider buying")
            elif macd < signal_line and macd_prev > signal_line_prev:
                signals.append("MACD: Bearish crossover - Consider selling")
            
            # Volume analysis
            current_volume = float(self.stock_data['Volume'].iloc[-1])
            avg_volume = float(self.stock_data['Volume'].rolling(window=20).mean().iloc[-1])
            
            if current_volume > avg_volume * 1.5:
                signals.append("VOLUME: Significantly higher than average - Strong price movement")
            elif current_volume < avg_volume * 0.5:
                signals.append("VOLUME: Significantly lower than average - Weak price movement")
            
            # Bollinger Bands analysis
            bb_upper = float(self.stock_data['BB_upper'].iloc[-1])
            bb_lower = float(self.stock_data['BB_lower'].iloc[-1])
            
            if current_price > bb_upper:
                signals.append("BOLLINGER: Price above upper band - Potential overbought")
            elif current_price < bb_lower:
                signals.append("BOLLINGER: Price below lower band - Potential oversold")
            
            # Get previous values for crossover calculations
            ma5_prev = float(self.stock_data['MA5'].iloc[-2])
            ma20_prev = float(self.stock_data['MA20'].iloc[-2])
            macd_prev = float(self.stock_data['MACD'].iloc[-2])
            signal_line_prev = float(self.stock_data['Signal_Line'].iloc[-2])
            
            # Format the signals
            signals_text = "\nTechnical Analysis Signals:\n"
            signals_text += "\nTrend Signals:\n"
            signals_text += "\n".join([s for s in signals if "TREND" in s])
            
            signals_text += "\n\nPrice Level Signals:\n"
            signals_text += "\n".join([s for s in signals if any(x in s for x in ["SUPPORT", "RESISTANCE", "BOLLINGER"])])
            
            signals_text += "\n\nMomentum Signals:\n"
            signals_text += "\n".join([s for s in signals if any(x in s for x in ["RSI", "MACD", "CROSSOVER"])])
            
            signals_text += "\n\nVolume Signals:\n"
            signals_text += "\n".join([s for s in signals if "VOLUME" in s])
            
            return signals_text
            
        except Exception as e:
            return f"Error generating trading signals: {str(e)}\nTry selecting a longer date range for better signal generation."

    def fetch_and_predict(self, stock_name, start_date, end_date):
        """Fetch data and make predictions"""
        if stock_name not in self.nse_stocks:
            return "Invalid stock selection!", None, None, None, None, None, None
        
        try:
            # Validate dates
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            if start_date >= end_date:
                return "Start date must be before end date!", None, None, None, None, None, None
            
            # Limit the maximum date range to 5 years
            if (end_date - start_date).days > 1825:  # 5 years
                return "Please select a date range of 5 years or less!", None, None, None, None, None, None
            
            # Fetch data with additional historical data for feature calculation
            start_date_with_buffer = (start_date - pd.Timedelta(days=200)).strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            ticker = self.nse_stocks[stock_name]
            self.selected_stock = stock_name
            self.stock_data = yf.download(ticker, start=start_date_with_buffer, end=end_date_str)
            
            if self.stock_data.empty:
                return "No data available for the selected period!", None, None, None, None, None, None
            
            # Calculate technical indicators
            # Moving Averages
            self.stock_data['MA200'] = self.stock_data['Close'].rolling(window=200, min_periods=1).mean()
            self.stock_data['MA30'] = self.stock_data['Close'].rolling(window=30, min_periods=1).mean()
            self.stock_data['MA20'] = self.stock_data['Close'].rolling(window=20, min_periods=1).mean()
            self.stock_data['MA5'] = self.stock_data['Close'].rolling(window=5, min_periods=1).mean()
            
            # Bollinger Bands (20-day)
            rolling_mean = self.stock_data['Close'].rolling(window=20).mean()
            rolling_std = self.stock_data['Close'].rolling(window=20).std()
            self.stock_data['BB_upper'] = rolling_mean + (rolling_std * 2)
            self.stock_data['BB_middle'] = rolling_mean
            self.stock_data['BB_lower'] = rolling_mean - (rolling_std * 2)
            
            # Support and Resistance levels
            self.stock_data['Support'] = self.stock_data['Low'].rolling(window=20, min_periods=1).min()
            self.stock_data['Resistance'] = self.stock_data['High'].rolling(window=20, min_periods=1).max()
            
            # Fibonacci Retracement Levels
            high = self.stock_data['High'].rolling(window=20, min_periods=1).max()
            low = self.stock_data['Low'].rolling(window=20, min_periods=1).min()
            diff = high - low
            self.stock_data['Fib_0.236'] = high - (diff * 0.236)
            self.stock_data['Fib_0.382'] = high - (diff * 0.382)
            self.stock_data['Fib_0.5'] = high - (diff * 0.5)
            self.stock_data['Fib_0.618'] = high - (diff * 0.618)
            self.stock_data['Fib_0.786'] = high - (diff * 0.786)
            
            # RSI
            delta = self.stock_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            self.stock_data['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = self.stock_data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = self.stock_data['Close'].ewm(span=26, adjust=False).mean()
            self.stock_data['MACD'] = exp1 - exp2
            self.stock_data['Signal_Line'] = self.stock_data['MACD'].ewm(span=9, adjust=False).mean()
            
            # Prepare features and target
            features_df = self.prepare_features(self.stock_data)
            
            # Trim the buffer period
            features_df = features_df[features_df.index >= start_date]
            self.stock_data = self.stock_data[self.stock_data.index >= start_date]
            
            if len(features_df) < 2:
                return "Insufficient data for analysis!", None, None, None, None, None, None
            
            # Prepare training data
            X = features_df.iloc[:-1]  # All rows except last
            y = self.stock_data['Close'].iloc[1:]  # All rows except first
            
            if len(X) < 30:  # Additional check for minimum required data
                return "Insufficient data for analysis! Please select a longer date range.", None, None, None, None, None, None
            
            # Train model
            try:
                self.train_model(X, y)
            except Exception as e:
                return f"Error training model: {str(e)}", None, None, None, None, None, None
            
            # Make future predictions
            try:
                future_pred_df = self.predict_future(self.stock_data)
            except Exception as e:
                return f"Error making predictions: {str(e)}", None, None, None, None, None, None
            
            # Create plots
            try:
                fig_price = self.create_price_plot(future_pred_df)
                fig_volume = self.create_volume_plot()
                fig_prediction = self.create_prediction_plot(future_pred_df)
            except Exception as e:
                print(f"Debug - Stock Data Columns: {self.stock_data.columns}")  # Debug print
                return f"Error creating plots: {str(e)}", None, None, None, None, None, None
            
            # Calculate metrics
            try:
                metrics = self.calculate_metrics(future_pred_df)
            except Exception as e:
                return f"Error calculating metrics: {str(e)}", None, None, None, None, None, None
            
            # Save all data files
            try:
                os.makedirs('data', exist_ok=True)
                
                # Save raw data
                raw_data_path = f'data/{stock_name.replace(" ", "_")}_raw_data.csv'
                self.stock_data.to_csv(raw_data_path)
                
                # Save analyzed data
                analyzed_data_path = f'data/{stock_name.replace(" ", "_")}_analyzed_data.csv'
                self.stock_data.to_csv(analyzed_data_path)
                
                # Save predictions
                predictions_path = f'data/{stock_name.replace(" ", "_")}_predictions.csv'
                future_pred_df.to_csv(predictions_path, index=False)
                
                # Create ZIP file immediately after analysis
                zip_file, status_msg = self.create_and_download_zip(stock_name)
                
                return ("Analysis completed successfully!", 
                        self.create_technical_analysis_plots(future_pred_df),
                        self.create_volume_analysis(),
                        self.create_momentum_indicators(),
                        metrics,
                        zip_file,    # File path first
                        status_msg)  # Status message second
                
            except Exception as e:
                print(f"Error saving files: {str(e)}")
                return f"Error saving files: {str(e)}", None, None, None, None, None, None
            
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            return f"Error during analysis: {str(e)}", None, None, None, None, None, None

    def create_technical_analysis_plots(self, future_pred_df):
        """Create comprehensive technical analysis plots"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{self.selected_stock} Price Analysis', 'Moving Averages'),
            row_heights=[0.7, 0.3]
        )
        
        # Main candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=self.stock_data.index,
                open=self.stock_data['Open'],
                high=self.stock_data['High'],
                low=self.stock_data['Low'],
                close=self.stock_data['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add Moving Averages
        ma_colors = {
            'MA5': 'rgba(255,255,0,0.7)',
            'MA20': 'rgba(0,255,0,0.7)',
            'MA30': 'rgba(255,165,0,0.7)',
            'MA200': 'rgba(255,0,0,0.7)'
        }
        
        for ma, color in ma_colors.items():
            fig.add_trace(
                go.Scatter(
                    x=self.stock_data.index,
                    y=self.stock_data[ma],
                    name=ma,
                    line=dict(color=color, width=1)
                ),
                row=2, col=1
            )
        
        # Add Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=self.stock_data.index,
                y=self.stock_data['BB_upper'],
                name='BB Upper',
                line=dict(color='gray', dash='dash'),
                opacity=0.3
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.stock_data.index,
                y=self.stock_data['BB_lower'],
                name='BB Lower',
                line=dict(color='gray', dash='dash'),
                fill='tonexty',
                opacity=0.3
            ),
            row=1, col=1
        )
        
        # Add Support and Resistance
        fig.add_trace(
            go.Scatter(
                x=self.stock_data.index,
                y=self.stock_data['Support'],
                name='Support',
                line=dict(color='green', dash='dot')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.stock_data.index,
                y=self.stock_data['Resistance'],
                name='Resistance',
                line=dict(color='red', dash='dot')
            ),
            row=1, col=1
        )
        
        # Add future predictions
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(future_pred_df['Date']),
                y=future_pred_df['Predicted_Close'],
                name='Prediction',
                line=dict(color='purple', dash='dash')
            ),
            row=1, col=1
        )
        
        # Update layout
        fig.update_layout(
            title_text=f"Technical Analysis for {self.selected_stock}",
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            template='plotly_dark'
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
import gradio as gr
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from model_training import prepare_training_data, train_xgboost_model, predict_test_data

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

# Create Gradio interface
with gr.Blocks(title="ML Pipeline") as app:
    gr.Markdown("# Machine Learning Pipeline with Step-by-Step Execution")
    
    pipeline = MLPipeline()
    
    with gr.Tab("1. Data Upload"):
        with gr.Row():
            train_input = gr.File(label="Upload Training CSV")
            test_input = gr.File(label="Upload Test CSV")
        upload_btn = gr.Button("Upload Data")
        upload_output = gr.Textbox(label="Upload Status")
    
    with gr.Tab("2. Data Cleaning"):
        clean_btn = gr.Button("Clean and Impute Data")
        clean_output = gr.Textbox(label="Cleaning Status")
    
    with gr.Tab("3. EDA"):
        eda_btn = gr.Button("Perform EDA")
        eda_output = gr.Textbox(label="EDA Status")
        with gr.Row():
            dist_plot = gr.Plot(label="Feature Distributions")
            corr_plot = gr.Plot(label="Correlation Heatmap")
            missing_plot = gr.Plot(label="Missing Values")
    
    with gr.Tab("4. Feature Engineering"):
        fe_btn = gr.Button("Engineer Features")
        fe_output = gr.Textbox(label="Feature Engineering Status")
    
    with gr.Tab("5. Hyperparameter Tuning"):
        tune_btn = gr.Button("Tune Hyperparameters")
        tune_output = gr.Textbox(label="Tuning Status")
        optimization_plot = gr.Plot(label="Optimization History")
    
    with gr.Tab("6. Training and Prediction"):
        train_btn = gr.Button("Train Model and Generate Predictions")
        train_output = gr.Textbox(label="Training Status")
        with gr.Row():
            feature_importance_plot = gr.Plot(label="Feature Importance")
            validation_plot = gr.Plot(label="Actual vs Predicted")
            predictions_dist_plot = gr.Plot(label="Predictions Distribution")
        metrics_output = gr.Textbox(label="Model Metrics and Best Parameters", lines=10)
        predictions_output = gr.File(label="Download Predictions CSV")
    
    # Set up button clicks
    upload_btn.click(
        fn=pipeline.upload_data,
        inputs=[train_input, test_input],
        outputs=[upload_output]
    )
    
    clean_btn.click(
        fn=pipeline.clean_and_impute,
        inputs=[],
        outputs=[clean_output]
    )
    
    eda_btn.click(
        fn=pipeline.perform_eda,
        inputs=[],
        outputs=[eda_output, dist_plot, corr_plot, missing_plot]
    )
    
    fe_btn.click(
        fn=pipeline.engineer_features,
        inputs=[],
        outputs=[fe_output]
    )
    
    tune_btn.click(
        fn=pipeline.tune_hyperparameters,
        inputs=[],
        outputs=[tune_output, optimization_plot]
    )
    
    train_btn.click(
        fn=pipeline.train_and_predict,
        inputs=[],
        outputs=[
            train_output,
            feature_importance_plot,
            validation_plot,
            predictions_dist_plot,
            metrics_output,
            predictions_output
        ]
    )

if __name__ == "__main__":
    app.launch(share=True) 
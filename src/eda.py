import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    """Load the cleaned data files"""
    data_dir = 'data'
    train_df = pd.read_csv(os.path.join(data_dir, 'clean_Train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'clean_Test.csv'))
    return train_df, test_df

def basic_analysis(df, title):
    """Print basic analysis of the dataframe"""
    print(f"\n{'-'*20} {title} {'-'*20}")
    print("\nDataset Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)
    print("\nBasic Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())

def plot_numerical_distributions(df, output_dir):
    """Plot distributions of numerical variables"""
    os.makedirs(output_dir, exist_ok=True)
    
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    for col in numerical_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(os.path.join(output_dir, f'{col}_distribution.png'))
        plt.close()

def plot_categorical_distributions(df, output_dir):
    """Plot distributions of categorical variables"""
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, x=col)
        plt.xticks(rotation=45)
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{col}_distribution.png'))
        plt.close()

def correlation_analysis(df, output_dir):
    """Analyze correlations between numerical variables"""
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()

def analyze_outlet_size(df):
    """Analyze Outlet_Size distribution with percentages"""
    size_dist = df['Outlet_Size'].value_counts()
    size_pct = df['Outlet_Size'].value_counts(normalize=True) * 100
    
    print("\nOutlet Size Distribution:")
    print("\nCounts:")
    print(size_dist)
    print("\nPercentages:")
    print(size_pct.round(2), "%")
    
    # Average sales by outlet size (if available in the dataset)
    if 'Item_Outlet_Sales' in df.columns:
        print("\nAverage Sales by Outlet Size:")
        print(df.groupby('Outlet_Size')['Item_Outlet_Sales'].mean().round(2))

def main():
    # Create output directory for plots
    output_dir = 'eda_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    train_df, test_df = load_data()
    
    # Basic analysis
    basic_analysis(train_df, "Training Data Analysis")
    basic_analysis(test_df, "Test Data Analysis")
    
    # Generate plots for training data
    print("\nGenerating plots for training data...")
    plot_numerical_distributions(train_df, output_dir)
    plot_categorical_distributions(train_df, output_dir)
    correlation_analysis(train_df, output_dir)
    
    # Additional insights
    print("\nUnique values in categorical columns (Training Data):")
    categorical_cols = train_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\n{col}:")
        print(train_df[col].value_counts())
    
    # Add outlet size analysis
    print("\nAnalyzing Outlet Size Distribution...")
    analyze_outlet_size(train_df)
    
    print("\nEDA completed! Plots have been saved in the 'eda_plots' directory.")

if __name__ == "__main__":
    main() 
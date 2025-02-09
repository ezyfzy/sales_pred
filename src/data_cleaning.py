import pandas as pd
import numpy as np
import os

def standardize_fat_content(fat_content):
    """Standardize fat content labels"""
    fat_content = str(fat_content).lower()
    if fat_content in ['low fat', 'lf']:
        return 'Low Fat'
    elif fat_content in ['reg', 'regular']:
        return 'Regular'
    return fat_content

def clean_data(df):
    """Clean and preprocess the data"""
    # Create a copy to avoid modifying original data
    cleaned_df = df.copy()
    
    # Standardize Item_Fat_Content
    cleaned_df['Item_Fat_Content'] = cleaned_df['Item_Fat_Content'].apply(standardize_fat_content)
    
    # Handle missing values
    # Fill missing Item_Weight with median weight for that Item_Type
    weight_median = cleaned_df.groupby('Item_Type')['Item_Weight'].transform('median')
    cleaned_df['Item_Weight'].fillna(weight_median, inplace=True)
    
    # Fill missing Outlet_Size based on Outlet_Type mode
    outlet_size_mode = cleaned_df.groupby('Outlet_Type')['Outlet_Size'].transform(lambda x: x.mode()[0])
    cleaned_df['Outlet_Size'].fillna(outlet_size_mode, inplace=True)
    
    # Replace 0 visibility with mean visibility of that product type
    zero_visibility_mask = cleaned_df['Item_Visibility'] == 0
    visibility_means = cleaned_df.groupby('Item_Type')['Item_Visibility'].transform('mean')
    cleaned_df.loc[zero_visibility_mask, 'Item_Visibility'] = visibility_means[zero_visibility_mask]
    
    return cleaned_df

def main():
    # Read the data
    try:
        # Create data directory if it doesn't exist
        data_dir = 'data'
        os.makedirs(data_dir, exist_ok=True)
        
        train_df = pd.read_csv(os.path.join(data_dir, 'Train.csv'))
        test_df = pd.read_csv(os.path.join(data_dir, 'Test.csv'))
        
        # Clean both datasets
        cleaned_train = clean_data(train_df)
        cleaned_test = clean_data(test_df)
        
        # Save cleaned datasets with 'clean_' prefix
        cleaned_train.to_csv(os.path.join(data_dir, 'clean_Train.csv'), index=False)
        cleaned_test.to_csv(os.path.join(data_dir, 'clean_Test.csv'), index=False)
        
        print("Data cleaning completed successfully!")
        print(f"Cleaned files saved as 'clean_Train.csv' and 'clean_Test.csv' in {data_dir}")
        
        # Print summary of changes
        print("\nSummary of cleaning operations:")
        print(f"Train data shape: {cleaned_train.shape}")
        print(f"Test data shape: {cleaned_test.shape}")
        print("\nMissing values in cleaned train data:")
        print(cleaned_train.isnull().sum())
        print("\nMissing values in cleaned test data:")
        print(cleaned_test.isnull().sum())
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the data files are in the correct location: '../data/Train.csv' and '../data/Test.csv'")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 
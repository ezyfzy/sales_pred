import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def load_data():
    """Load the cleaned data files"""
    data_dir = 'data'
    train_df = pd.read_csv(os.path.join(data_dir, 'clean_Train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'clean_Test.csv'))
    return train_df, test_df

def create_new_features(df):
    """Create new features from existing ones"""
    # Create copy to avoid modifying original data
    df = df.copy()
    
    # 1. Item Type Category
    # Group similar item types into broader categories
    item_category_map = {
        'Foods': ['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Snack Foods', 
                 'Breads', 'Breakfast', 'Seafood', 'Starchy Foods', 'Others', 'Hard Drinks'],
        'Non-Foods': ['Health and Hygiene', 'Household', 'Baking Goods']
    }
    
    df['Item_Category'] = df['Item_Type'].map(
        {item: category for category, items in item_category_map.items() 
         for item in items}
    )
    
    # 2. Price Level Feature
    df['Price_Level'] = pd.qcut(df['Item_MRP'], q=4, labels=['Budget', 'Economy', 'Mid_Range', 'Premium'])
    
    # 3. Store Age Feature
    df['Store_Age'] = 2013 - df['Outlet_Establishment_Year']
    
    # 4. Item Identifier Features
    df['Item_Type_Code'] = df['Item_Identifier'].str[:2]
    df['Item_Code'] = df['Item_Identifier'].str[2:]
    
    # 5. Price per Weight
    df['Price_Per_Weight'] = df['Item_MRP'] / df['Item_Weight']
    
    # 6. Outlet Location and Size Combined
    df['Outlet_Location_Size'] = df['Outlet_Location_Type'] + '_' + df['Outlet_Size']
    
    return df

def encode_categorical_features(train_df, test_df):
    """Encode categorical features"""
    # Create copies
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    # Features to encode
    categorical_features = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size',
                          'Outlet_Location_Type', 'Outlet_Type', 'Item_Category',
                          'Price_Level', 'Item_Type_Code', 'Outlet_Location_Size']
    
    # Initialize dictionary to store label encoders
    label_encoders = {}
    
    # Encode each categorical feature
    for feature in categorical_features:
        label_encoders[feature] = LabelEncoder()
        # Fit on both train and test to include all categories
        all_values = pd.concat([train_df[feature], test_df[feature]])
        label_encoders[feature].fit(all_values)
        
        # Transform train and test
        train_encoded[feature] = label_encoders[feature].transform(train_df[feature])
        test_encoded[feature] = label_encoders[feature].transform(test_df[feature])
    
    return train_encoded, test_encoded, label_encoders

def scale_numerical_features(train_df, test_df):
    """Scale numerical features"""
    numerical_features = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 
                         'Store_Age', 'Price_Per_Weight']
    
    # Create copies
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    
    # Scale each numerical feature
    for feature in numerical_features:
        mean = train_df[feature].mean()
        std = train_df[feature].std()
        
        train_scaled[feature] = (train_df[feature] - mean) / std
        test_scaled[feature] = (test_df[feature] - mean) / std
    
    return train_scaled, test_scaled

def prepare_features(train_df, test_df):
    """Prepare features for modeling"""
    # Create new features
    print("Creating new features...")
    train_featured = create_new_features(train_df)
    test_featured = create_new_features(test_df)
    
    # Encode categorical features
    print("Encoding categorical features...")
    train_encoded, test_encoded, encoders = encode_categorical_features(train_featured, test_featured)
    
    # Scale numerical features
    print("Scaling numerical features...")
    train_prepared, test_prepared = scale_numerical_features(train_encoded, test_encoded)
    
    return train_prepared, test_prepared

def main():
    # Load data
    print("Loading data...")
    train_df, test_df = load_data()
    
    # Prepare features
    train_prepared, test_prepared = prepare_features(train_df, test_df)
    
    # Save prepared datasets
    print("Saving prepared datasets...")
    train_prepared.to_csv('data/prepared_train.csv', index=False)
    test_prepared.to_csv('data/prepared_test.csv', index=False)
    
    # Print feature information
    print("\nPrepared training data shape:", train_prepared.shape)
    print("Prepared test data shape:", test_prepared.shape)
    print("\nFeatures in prepared data:", train_prepared.columns.tolist())
    
    print("\nFeature engineering completed!")

if __name__ == "__main__":
    main() 
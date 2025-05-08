import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

def main():
    # Path to the CSV with extracted pixel values and training data
    csv_path = r'working/training-data/training-data-with-pixels.csv'
    
    # Load the data into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Define the feature columns and target variable
    #feature_cols = ['S2_LT_B5', 'S2_LT_B8', 'S2_LT_B12', 'S2_LT_B2', 'S2_LT_B3', 'S2_LT_B4']
    lt_false_color_feature_cols = ['S2_LT_B5', 'S2_LT_B8', 'S2_LT_B12']
    lt_true_color_feature_cols = ['S2_LT_B2', 'S2_LT_B3', 'S2_LT_B4']
    at_true_color_feature_cols = ['S2_AT_B2', 'S2_AT_B3', 'S2_AT_B4']
    all_feature_cols = lt_false_color_feature_cols + lt_true_color_feature_cols + at_true_color_feature_cols
    # lt_feature_cols = lt_false_color_feature_cols + lt_true_color_feature_cols

    X = df[all_feature_cols].values
    y = df['FeatType'].values
    
    # Encode the categorical target labels into numeric values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Find the index of the rocky reef class
    rocky_reef_index = np.where(le.classes_ == 'Rocky reef')[0][0]
    print(f"Rocky reef index: {rocky_reef_index}, Class name: {le.classes_[rocky_reef_index]}")
    
    # Count samples per class
    unique_classes, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique_classes, counts))
    print("Class distribution before balancing:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Create and train the Random Forest classifier without class weights
    rf = RandomForestClassifier(
        n_estimators=50, 
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    # Evaluate the model on the test set
    y_pred = rf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save the trained model and the label encoder to disk
    model_dir = r'working/training-data'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'random_forest_model.pkl')
    encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
    joblib.dump(rf, model_path)
    joblib.dump(le, encoder_path)
    
    print(f"Random Forest model saved to: {model_path}")
    print(f"Label Encoder saved to: {encoder_path}")

if __name__ == '__main__':
    main()

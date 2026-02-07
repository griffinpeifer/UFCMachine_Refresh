"""
Script to experiment with different neural network architectures for UFC fight prediction.
This allows you to easily test different configurations and compare results.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def load_data():
    """Load and prepare the training data."""
    df_matches = pd.read_csv('training_model.csv')
    
    X = df_matches.iloc[:, 1:]  # All columns except 'result'
    y = df_matches.iloc[:, 0]   # 'result' column
    
    # Handle any NaN values
    X = X.fillna(0)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def create_model(architecture_config, input_dim):
    """
    Create a neural network model based on architecture configuration.
    
    Args:
        architecture_config: Dictionary with keys:
            - 'layers': List of layer sizes (e.g., [32, 64, 32])
            - 'dropout': Dropout rate (0.0 to 1.0), or None to disable
            - 'l2_reg': L2 regularization strength (e.g., 0.01), or None to disable
            - 'activation': Activation function (default: 'relu')
        input_dim: Number of input features
    
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    
    layers = architecture_config.get('layers', [32, 32])
    dropout = architecture_config.get('dropout', None)
    l2_reg = architecture_config.get('l2_reg', 0.01)
    activation = architecture_config.get('activation', 'relu')
    
    # Add input layer
    reg = regularizers.L2(l2_reg) if l2_reg else None
    model.add(Dense(
        layers[0],
        input_dim=input_dim,
        activation=activation,
        kernel_regularizer=reg
    ))
    
    if dropout:
        model.add(Dropout(dropout))
    
    # Add hidden layers
    for layer_size in layers[1:]:
        reg = regularizers.L2(l2_reg) if l2_reg else None
        model.add(Dense(
            layer_size,
            activation=activation,
            kernel_regularizer=reg
        ))
        if dropout:
            model.add(Dropout(dropout))
    
    # Add output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['binary_accuracy']
    )
    
    return model

def train_and_evaluate(model, X_train, X_test, y_train, y_test, epochs=200, batch_size=64):
    """
    Train a model and return evaluation metrics.
    
    Returns:
        Dictionary with training history and test metrics
    """
    # Split training data for validation
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=0
    )
    
    # Train model
    history = model.fit(
        X_train_fit,
        y_train_fit,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[early_stopping]
    )
    
    # Evaluate on test set
    test_results = model.evaluate(X_test, y_test, verbose=0)
    
    return {
        'test_accuracy': test_results[1],
        'test_loss': test_results[0],
        'epochs_trained': len(history.history['loss']),
        'history': history.history
    }

def experiment_with_architectures():
    """
    Test multiple architecture configurations and compare results.
    """
    print("Loading data...")
    X_train, X_test, y_train, y_test, scaler = load_data()
    print(f"Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Input features: {X_train.shape[1]}\n")
    
    # Define different architecture configurations to test
    architectures = [
        {
            'name': 'Small (2 layers)',
            'config': {
                'layers': [32, 16],
                'dropout': None,
                'l2_reg': 0.01,
                'activation': 'relu'
            }
        },
        {
            'name': 'Medium (3 layers)',
            'config': {
                'layers': [64, 32, 16],
                'dropout': None,
                'l2_reg': 0.01,
                'activation': 'relu'
            }
        },
        {
            'name': 'Large (4 layers)',
            'config': {
                'layers': [128, 64, 32, 16],
                'dropout': None,
                'l2_reg': 0.01,
                'activation': 'relu'
            }
        },
        {
            'name': 'Deep (5 layers)',
            'config': {
                'layers': [64, 64, 64, 32, 16],
                'dropout': None,
                'l2_reg': 0.01,
                'activation': 'relu'
            }
        },
        {
            'name': 'With Dropout',
            'config': {
                'layers': [64, 32, 16],
                'dropout': 0.3,
                'l2_reg': 0.01,
                'activation': 'relu'
            }
        },
        {
            'name': 'No Regularization',
            'config': {
                'layers': [64, 32, 16],
                'dropout': None,
                'l2_reg': None,
                'activation': 'relu'
            }
        },
        {
            'name': 'Wide Layers',
            'config': {
                'layers': [128, 128, 64],
                'dropout': None,
                'l2_reg': 0.01,
                'activation': 'relu'
            }
        },
        {
            'name': 'Narrow Layers',
            'config': {
                'layers': [16, 8, 4],
                'dropout': None,
                'l2_reg': 0.01,
                'activation': 'relu'
            }
        },
    ]
    
    results = []
    
    print("=" * 80)
    print("EXPERIMENTING WITH DIFFERENT ARCHITECTURES")
    print("=" * 80)
    print()
    
    for arch in architectures:
        print(f"Testing: {arch['name']}")
        print(f"  Architecture: {arch['config']['layers']}")
        print(f"  Dropout: {arch['config']['dropout']}")
        print(f"  L2 Regularization: {arch['config']['l2_reg']}")
        
        # Create and train model
        model = create_model(arch['config'], X_train.shape[1])
        metrics = train_and_evaluate(model, X_train, X_test, y_train, y_test)
        
        # Store results
        results.append({
            'name': arch['name'],
            'architecture': arch['config']['layers'],
            'dropout': arch['config']['dropout'],
            'l2_reg': arch['config']['l2_reg'],
            'test_accuracy': metrics['test_accuracy'],
            'test_loss': metrics['test_loss'],
            'epochs_trained': metrics['epochs_trained'],
            'model': model  # Save model for later use
        })
        
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f} ({metrics['test_accuracy']*100:.2f}%)")
        print(f"  Test Loss: {metrics['test_loss']:.4f}")
        print(f"  Epochs Trained: {metrics['epochs_trained']}")
        print()
    
    # Summary
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    # Sort by test accuracy
    results_sorted = sorted(results, key=lambda x: x['test_accuracy'], reverse=True)
    
    print(f"{'Rank':<6} {'Architecture':<25} {'Test Accuracy':<15} {'Test Loss':<12} {'Epochs':<8}")
    print("-" * 80)
    for i, r in enumerate(results_sorted, 1):
        arch_str = str(r['architecture'])
        print(f"{i:<6} {r['name']:<25} {r['test_accuracy']*100:>6.2f}%        {r['test_loss']:>6.4f}     {r['epochs_trained']:<8}")
    
    print()
    print(f"Best Model: {results_sorted[0]['name']}")
    print(f"  Architecture: {results_sorted[0]['architecture']}")
    print(f"  Test Accuracy: {results_sorted[0]['test_accuracy']*100:.2f}%")
    
    # Save best model
    best_model = results_sorted[0]['model']
    best_model.save('best_ufc_model.h5')
    print(f"\nBest model saved as 'best_ufc_model.h5'")
    
    return results_sorted

if __name__ == "__main__":
    results = experiment_with_architectures()

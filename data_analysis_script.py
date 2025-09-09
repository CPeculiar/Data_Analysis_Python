"""
Data Analysis with Pandas and Visualization with Matplotlib
Assignment Solution

This script demonstrates:
- Loading and exploring datasets using pandas
- Basic data analysis and statistics
- Creating visualizations with matplotlib
- Data insights and findings
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
import seaborn as sns

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_and_explore_data():
    """Task 1: Load and Explore the Dataset"""
    print("=== TASK 1: LOAD AND EXPLORE DATASET ===")
    
    try:
        # Load the Iris dataset
        iris_data = load_iris()
        
        # Create DataFrame
        df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
        df['species'] = iris_data.target_names[iris_data.target]
        
        print("Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        
        # Display first few rows
        print("\nFirst 5 rows of the dataset:")
        print(df.head())
        
        # Explore dataset structure
        print("\nDataset Info:")
        df.info()
        print("\nData Types:")
        print(df.dtypes)
        print("\nMissing Values:")
        print(df.isnull().sum())
        
        # Check for missing values and clean if necessary
        if df.isnull().sum().sum() > 0:
            print("Missing values found. Cleaning dataset...")
            df = df.dropna()
        else:
            print("No missing values found. Dataset is clean!")
        
        print(f"Final dataset shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def basic_data_analysis(df):
    """Task 2: Basic Data Analysis"""
    print("\n=== TASK 2: BASIC DATA ANALYSIS ===")
    
    # Basic statistics for numerical columns
    print("Basic Statistics:")
    print(df.describe())
    
    # Group by species and compute mean
    print("\nMean values by species:")
    species_means = df.groupby('species').mean()
    print(species_means)
    
    # Additional analysis
    print("\nSpecies distribution:")
    species_counts = df['species'].value_counts()
    print(species_counts)
    
    # Correlation analysis
    print("\nCorrelation Matrix:")
    correlation_matrix = df.select_dtypes(include=[np.number]).corr()
    print(correlation_matrix)
    
    return species_means, species_counts, correlation_matrix

def create_visualizations(df, species_means):
    """Task 3: Data Visualization"""
    print("\n=== TASK 3: DATA VISUALIZATION ===")
    
    features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    
    # 1. Line Chart - Trends Over Index
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length', alpha=0.7)
    plt.plot(df.index, df['petal length (cm)'], label='Petal Length', alpha=0.7)
    plt.title('Sepal and Petal Length Trends Across Samples', fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index')
    plt.ylabel('Length (cm)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 2. Bar Chart - Average Measurements by Species
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, feature in enumerate(features):
        ax = axes[i//2, i%2]
        species_means[feature].plot(kind='bar', ax=ax, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax.set_title(f'Average {feature.title()} by Species', fontweight='bold')
        ax.set_ylabel('Length (cm)')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # 3. Histogram - Distribution of Numerical Columns
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, feature in enumerate(features):
        ax = axes[i//2, i%2]
        ax.hist(df[feature], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_title(f'Distribution of {feature.title()}', fontweight='bold')
        ax.set_xlabel(f'{feature.title()}')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 4. Scatter Plot - Relationship Between Variables
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot 1: Sepal Length vs Sepal Width
    for species in df['species'].unique():
        species_data = df[df['species'] == species]
        axes[0].scatter(species_data['sepal length (cm)'], species_data['sepal width (cm)'], 
                       label=species, alpha=0.7, s=50)
    
    axes[0].set_title('Sepal Length vs Sepal Width', fontweight='bold')
    axes[0].set_xlabel('Sepal Length (cm)')
    axes[0].set_ylabel('Sepal Width (cm)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Scatter plot 2: Petal Length vs Petal Width
    for species in df['species'].unique():
        species_data = df[df['species'] == species]
        axes[1].scatter(species_data['petal length (cm)'], species_data['petal width (cm)'], 
                       label=species, alpha=0.7, s=50)
    
    axes[1].set_title('Petal Length vs Petal Width', fontweight='bold')
    axes[1].set_xlabel('Petal Length (cm)')
    axes[1].set_ylabel('Petal Width (cm)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def display_findings(df, species_means, species_counts, correlation_matrix):
    """Display key findings and observations"""
    print("\n=== KEY FINDINGS AND OBSERVATIONS ===")
    print("\n1. Dataset Overview:")
    print(f"   - Total samples: {len(df)}")
    print(f"   - Features: {len(df.columns)-1}")
    print(f"   - Species: {df['species'].nunique()}")
    
    print("\n2. Species Distribution:")
    for species, count in species_counts.items():
        print(f"   - {species}: {count} samples")
    
    print("\n3. Key Observations:")
    print(f"   - Largest average sepal length: {species_means['sepal length (cm)'].idxmax()}")
    print(f"   - Largest average petal length: {species_means['petal length (cm)'].idxmax()}")
    print(f"   - Strongest correlation: {correlation_matrix.abs().unstack().sort_values(ascending=False).iloc[1:2].index[0]}")
    
    print("\n4. Insights:")
    print("   - Virginica species generally has the largest measurements")
    print("   - Setosa species is clearly distinguishable by smaller petal measurements")
    print("   - Strong positive correlation between petal length and width")
    print("   - Dataset is well-balanced with equal samples per species")

def main():
    """Main function to run the complete analysis"""
    print("Data Analysis with Pandas and Visualization with Matplotlib")
    print("=" * 60)
    
    # Task 1: Load and explore data
    df = load_and_explore_data()
    if df is None:
        return
    
    # Task 2: Basic data analysis
    species_means, species_counts, correlation_matrix = basic_data_analysis(df)
    
    # Task 3: Create visualizations
    create_visualizations(df, species_means)
    
    # Display findings
    display_findings(df, species_means, species_counts, correlation_matrix)
    
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()
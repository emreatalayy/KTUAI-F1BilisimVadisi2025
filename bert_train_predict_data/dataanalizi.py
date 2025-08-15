# Comprehensive Data Analysis and Visualization for Dataset
# Columns: id, text, category, topic, emotion, urgency

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('fivethirtyeight')
sns.set(style="whitegrid")

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

def load_data(file_path):
    """Load dataset from file path"""
    # Try to detect file format based on extension
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide CSV, Excel, or JSON file.")
    
    return df

def basic_exploration(df):
    """Basic exploration of the dataset"""
    print("Dataset Overview:")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumn names:", df.columns.tolist())
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    print("\nSample data (first 5 rows):")
    print(df.head())
    
    print("\nBasic statistics for numerical columns:")
    print(df.describe().T)

def analyze_categorical_data(df):
    """Analyze categorical columns: category, topic, emotion, urgency"""
    categorical_cols = ['category', 'topic', 'emotion', 'urgency']
    available_cols = [col for col in categorical_cols if col in df.columns]
    
    print("\nCategorical Columns Analysis:")
    for col in available_cols:
        print(f"\n{col.title()} Distribution:")
        value_counts = df[col].value_counts()
        print(value_counts)
        print(f"Number of unique {col}s: {df[col].nunique()}")
        
        # Create horizontal bar chart
        plt.figure(figsize=(12, 8))
        sns.countplot(y=df[col], order=df[col].value_counts().index, palette='viridis')
        plt.title(f'Distribution of {col.title()}', fontsize=15)
        plt.xlabel('Count', fontsize=12)
        plt.ylabel(col.title(), fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{col}_distribution.png')
        plt.close()
        
        # Create pie chart for top categories
        plt.figure(figsize=(10, 10))
        top_categories = df[col].value_counts().nlargest(10)
        plt.pie(top_categories, labels=top_categories.index, autopct='%1.1f%%', startangle=90, 
                shadow=True, explode=[0.05]*len(top_categories), textprops={'fontsize': 12})
        plt.title(f'Top 10 {col.title()} Distribution', fontsize=15)
        plt.tight_layout()
        plt.savefig(f'{col}_pie_chart.png')
        plt.close()

def analyze_text_data(df):
    """Analyze text column content"""
    if 'text' not in df.columns:
        print("No 'text' column found in the dataset")
        return
    
    print("\nText Data Analysis:")
    
    # Text length statistics
    df['text_length'] = df['text'].astype(str).apply(len)
    df['word_count'] = df['text'].astype(str).apply(lambda x: len(str(x).split()))
    
    print("\nText length statistics:")
    print(df[['text_length', 'word_count']].describe())
    
    # Plot text length distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df['word_count'], kde=True, bins=50, color='skyblue')
    plt.title('Distribution of Word Count in Text', fontsize=15)
    plt.xlabel('Word Count', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig('text_length_distribution.png')
    plt.close()
    
    # Most common words
    stop_words = set(stopwords.words('english'))
    
    def extract_words(text):
        text = str(text).lower()
        words = re.findall(r'\b[a-z]{3,15}\b', text)
        return [word for word in words if word not in stop_words]
    
    all_words = []
    for text in df['text'].astype(str):
        all_words.extend(extract_words(text))
    
    word_freq = Counter(all_words)
    print("\nTop 20 most common words:")
    print(pd.DataFrame(word_freq.most_common(20), columns=['Word', 'Frequency']))
    
    # Create word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                          max_words=100, contour_width=3, contour_color='steelblue')
    wordcloud.generate(' '.join(all_words))
    
    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Text Data', fontsize=18)
    plt.tight_layout()
    plt.savefig('text_wordcloud.png')
    plt.close()

def analyze_relationships(df):
    """Analyze relationships between categorical variables"""
    categorical_cols = ['category', 'topic', 'emotion', 'urgency']
    available_cols = [col for col in categorical_cols if col in df.columns]
    
    if len(available_cols) < 2:
        print("Not enough categorical columns for relationship analysis")
        return
    
    print("\nRelationships between categorical variables:")
    
    # Create heatmaps for pairs of categorical variables
    for i in range(len(available_cols)):
        for j in range(i+1, len(available_cols)):
            col1, col2 = available_cols[i], available_cols[j]
            
            # Create contingency table
            contingency = pd.crosstab(df[col1], df[col2], normalize='index')
            
            plt.figure(figsize=(14, 10))
            sns.heatmap(contingency, annot=False, cmap="YlGnBu", fmt='.2f')
            plt.title(f'Relationship between {col1.title()} and {col2.title()}', fontsize=15)
            plt.tight_layout()
            plt.savefig(f'{col1}_{col2}_relationship.png')
            plt.close()
            
            # Print top combinations
            print(f"\nTop combinations of {col1} and {col2}:")
            combo_counts = df.groupby([col1, col2]).size().reset_index(name='count')
            combo_counts = combo_counts.sort_values('count', ascending=False).head(10)
            print(combo_counts)

def create_interactive_visualizations(df):
    """Create interactive visualizations using plotly"""
    categorical_cols = ['category', 'topic', 'emotion', 'urgency']
    available_cols = [col for col in categorical_cols if col in df.columns]
    
    if len(available_cols) == 0:
        print("No categorical columns for interactive visualizations")
        return
    
    # Create interactive bar chart for one categorical variable
    main_col = available_cols[0]
    
    # Fix: Create a proper DataFrame for plotting
    value_counts = df[main_col].value_counts()
    plot_df = pd.DataFrame({
        'category': value_counts.index,
        'count': value_counts.values
    })
    
    fig = px.bar(plot_df, 
                 x='category', y='count', 
                 title=f'Distribution of {main_col.title()}',
                 labels={'category': main_col.title(), 'count': 'Count'},
                 color='category',
                 height=600)
    fig.write_html(f'{main_col}_interactive_bar.html')
    
    # If we have text data, create a scatter plot of text length vs. a categorical variable
    if 'text' in df.columns and df['text'].dtype == 'object':
        df['text_length'] = df['text'].astype(str).apply(len)
        
        if len(available_cols) >= 1:
            fig = px.box(df, x=available_cols[0], y='text_length',
                         title=f'Text Length by {available_cols[0].title()}',
                         labels={available_cols[0]: available_cols[0].title(), 'text_length': 'Text Length'},
                         color=available_cols[0],
                         height=600)
            fig.write_html(f'text_length_by_{available_cols[0]}.html')
    
    # Create a sunburst chart if we have multiple categorical columns
    if len(available_cols) >= 2:
        fig = px.sunburst(df, path=available_cols[:3],  # Use up to 3 columns
                          title='Hierarchical View of Categories',
                          height=700)
        fig.write_html('category_hierarchy_sunburst.html')

def create_dashboard(df):
    """Create a simple dashboard of visualizations"""
    categorical_cols = ['category', 'topic', 'emotion', 'urgency']
    available_cols = [col for col in categorical_cols if col in df.columns]
    
    if len(available_cols) == 0:
        print("No categorical columns for dashboard")
        return
    
    # Create a subplot grid
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "bar"}, {"type": "pie"}],
               [{"type": "bar"}, {"type": "bar"}]],
        subplot_titles=(
            f'Distribution of {available_cols[0].title()}',
            'Top Categories',
            f'Average Text Length by {available_cols[0].title()}' if 'text' in df.columns else 'Distribution of Another Variable',
            'Frequency of Data'
        )
    )
    
    # Plot 1: Bar chart of first categorical variable
    counts = df[available_cols[0]].value_counts().nlargest(10)
    fig.add_trace(
        go.Bar(x=counts.index, y=counts.values, marker_color='skyblue'),
        row=1, col=1
    )
    
    # Plot 2: Pie chart of first categorical variable
    fig.add_trace(
        go.Pie(labels=counts.index, values=counts.values, hole=.3),
        row=1, col=2
    )
    
    # Plot 3: Bar chart of average text length by category (if text exists)
    if 'text' in df.columns:
        df['text_length'] = df['text'].astype(str).apply(len)
        avg_length = df.groupby(available_cols[0])['text_length'].mean().nlargest(10)
        fig.add_trace(
            go.Bar(x=avg_length.index, y=avg_length.values, marker_color='lightgreen'),
            row=2, col=1
        )
    elif len(available_cols) > 1:
        # If no text, use second categorical variable
        counts2 = df[available_cols[1]].value_counts().nlargest(10)
        fig.add_trace(
            go.Bar(x=counts2.index, y=counts2.values, marker_color='lightgreen'),
            row=2, col=1
        )
    
    # Plot 4: Count of records over time (if we have a date column)
    date_cols = df.select_dtypes(include=['datetime']).columns
    if len(date_cols) > 0:
        df['month'] = df[date_cols[0]].dt.to_period('M')
        time_counts = df.groupby('month').size()
        fig.add_trace(
            go.Bar(x=time_counts.index.astype(str), y=time_counts.values, marker_color='coral'),
            row=2, col=2
        )
    else:
        # If no date column, use overall count
        fig.add_trace(
            go.Bar(x=['Total Records'], y=[len(df)], marker_color='coral'),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=900,
        width=1200,
        title_text="Data Analysis Dashboard",
        showlegend=False
    )
    
    # Save dashboard
    fig.write_html('data_dashboard.html')

def main():
    """Main function to run all analyses"""
    # Ask user for file path
    file_path = input("Enter the path to your data file: ")
    
    try:
        # Load data
        df = load_data(file_path)
        
        # Run all analyses
        basic_exploration(df)
        analyze_categorical_data(df)
        analyze_text_data(df)
        analyze_relationships(df)
        create_interactive_visualizations(df)
        create_dashboard(df)
        
        print("\nAnalysis complete! All visualizations have been saved to the current directory.")
        print("Check the following files:")
        print("- PNG files for each visualization")
        print("- HTML files for interactive visualizations")
        print("- data_dashboard.html for a comprehensive dashboard")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
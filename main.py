import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from scipy import stats
from itertools import combinations

def load_and_merge(movies_path: str, credits_path: str) -> pd.DataFrame:
    """Load and merge TMDB movies and credits datasets."""
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    df = pd.merge(movies, credits, left_on='id', right_on='movie_id', how='inner')
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Convert release_date to datetime, extract year, and engineer genre features."""
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['year'] = df['release_date'].dt.year

    df['genre_list'] = df['genres'].apply(
        lambda x: [d['name'] for d in ast.literal_eval(x)]
    )
    df['primary_genre'] = df['genre_list'].apply(lambda lst: lst[0] if lst else None)
    df['genre_count'] = df['genre_list'].apply(len)

    df['runtime_bucket'] = pd.cut(
        df['runtime'],
        bins=[0, 90, 120, 150, np.inf],
        labels=['<=90', '91-120', '121-150', '150+']
    )
    return df

def print_all_pearson(df: pd.DataFrame, cols: list):
    """Compute & print Pearson r and p-value for every unique pair in cols."""
    print("\n=== Pairwise Pearson Correlations with p-values ===")
    for a, b in combinations(cols, 2):
        x, y = df[a], df[b]
        mask = x.notna() & y.notna()
        if mask.sum() < 2:
            continue
        r, p = stats.pearsonr(x[mask], y[mask])
        print(f"{a:12s} ↔ {b:12s} : r = {r:6.3f}, p = {p:.2e}")

def eda(df: pd.DataFrame):
    """Perform comprehensive exploratory data analysis."""
    # 1. Missing values and duplicates
    print("=== Missing Values per Column ===")
    print(df.isnull().sum())
    dup_count = df.duplicated(subset=['id']).sum()
    print(f"\n=== Duplicate Rows (by id) ===\n{dup_count}")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False)
    plt.title("Missing Data Heatmap")
    plt.show()

    # 2. Descriptive stats
    numeric = [
        'budget', 'revenue', 'popularity',
        'vote_count', 'vote_average', 'runtime', 'genre_count'
    ]
    print("\n=== Descriptive Statistics ===")
    print(df[numeric].describe())

    # 3. Boxplots for outlier detection
    plt.figure(figsize=(5, 5))
    df[numeric].boxplot()
    plt.title("Boxplots of Numeric Features")
    plt.xticks(rotation=45)
    plt.show()

    # 4. Separate histograms for each numeric variable
    for col in numeric:
        plt.figure(figsize=(6, 4))
        df[col].dropna().hist(bins=50)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    # 5. Pairplot (scatter‐matrix) of numeric features (sampled)
    sample_df = df[numeric].dropna().sample(min(len(df), 500))
    g = sns.pairplot(sample_df, corner=True)
    g.fig.suptitle("Pairwise Scatterplots (sampled)", y=1.02)
    plt.show()

    # 6. Correlation matrix & heatmap
    corr = df[numeric].corr()
    print("\n=== Correlation Matrix ===")
    print(corr)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Numeric Features")
    plt.show()

    # 6b. Detailed Pearson r & p-values
    print_all_pearson(df, numeric)

    # 7. Categorical distributions
    cats = ['primary_genre', 'original_language', 'runtime_bucket']
    fig, axes = plt.subplots(1, len(cats), figsize=(18, 5))
    for ax, var in zip(axes, cats):
        counts = df[var].value_counts().nlargest(10)
        counts.plot(kind='bar', ax=ax)
        ax.set_title(f'Top 10 {var.replace("_"," ").title()}')
        ax.set_xlabel(var.replace("_"," ").title())
        ax.set_ylabel("Count")
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

    # 8. Scatter plots with regression lines
    pairs = [
        ('budget', 'revenue'),
        ('popularity', 'revenue'),
        ('vote_average', 'revenue')
    ]
    for x, y in pairs:
        plt.figure(figsize=(8, 5))
        sns.regplot(
            x=x, y=y, data=df,
            scatter_kws={'alpha':0.5},
            line_kws={'color':'red'}
        )
        plt.title(f"{y.title()} vs {x.title()} with Regression Line")
        plt.xlabel(x.replace('_',' ').title())
        plt.ylabel(y.replace('_',' ').title())
        plt.show()

def hypothesis_tests(df: pd.DataFrame):
    """Run hypothesis tests with clear null/alternative statements."""
    clean = df[['budget', 'revenue']].dropna()
    r, p = stats.pearsonr(clean['budget'], clean['revenue'])
    print("\n--- Hypothesis Test 1: Budget vs Revenue ---")
    print("H0: ρ = 0 (no correlation)\nHa: ρ ≠ 0 (non-zero correlation)")
    print(f"Pearson r = {r:.3f}, p-value = {p:.3e}")
    print("→", "Reject H0" if p < 0.05 else "Fail to reject H0")

    median_rating = df['vote_average'].median()
    grp_high = df[df['vote_average'] > median_rating]['revenue'].dropna()
    grp_low  = df[df['vote_average'] <= median_rating]['revenue'].dropna()
    t_stat, p_val = stats.ttest_ind(grp_high, grp_low, equal_var=False)
    print("\n--- Hypothesis Test 2: High vs Low Rating Revenue ---")
    print("H0: μ_high = μ_low\nHa: μ_high ≠ μ_low")
    print(f"t-statistic = {t_stat:.3f}, p-value = {p_val:.3e}")
    print("→", "Reject H0" if p_val < 0.05 else "Fail to reject H0")

if __name__ == "__main__":
    MOVIES_CSV  = 'tmdb_5000_movies.csv'
    CREDITS_CSV = 'tmdb_5000_credits.csv'

    df = load_and_merge(MOVIES_CSV, CREDITS_CSV)
    df = preprocess(df)
    eda(df)
    hypothesis_tests(df)

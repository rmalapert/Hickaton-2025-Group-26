import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print(" CORRELATION ANALYSIS: X_test COLUMNS WITH MathScore")
print("="*80)

# Load data
print("\n📂 Loading data...")
X_train = pd.read_csv("X_train.csv", index_col=0)
X_test = pd.read_csv("X_test.csv", index_col=0)
y_train = pd.read_csv("y_train.csv", index_col=0)
print("✓ Files loaded successfully.")

# Merge X_train with y_train to calculate correlations
df_train = X_train.join(y_train, how='inner')

print(f"\n📊 Dataset Dimensions:")
print(f"   • X_train: {X_train.shape[0]:,} observations × {X_train.shape[1]} variables")
print(f"   • X_test:  {X_test.shape[0]:,} observations × {X_test.shape[1]} variables")
print(f"   • y_train: {y_train.shape[0]:,} observations")

# Filter valid scores (remove zeros)
df_valid = df_train[df_train['MathScore'] > 0].copy()
print(f"\n📈 Valid observations (MathScore > 0): {len(df_valid):,}")

# Calculate correlations for numeric columns only
print("\n🔍 Calculating correlations with MathScore...")
numeric_cols = df_valid.select_dtypes(include=np.number).columns.tolist()
numeric_cols = [col for col in numeric_cols if col != 'MathScore']

correlations = df_valid[numeric_cols + ['MathScore']].corr()['MathScore'].drop('MathScore')
correlations = correlations.sort_values(ascending=False)

# Remove NaN correlations
correlations = correlations.dropna()

print(f"✓ Correlations calculated for {len(correlations)} numeric columns.")

# Save full correlation results to CSV
correlation_df = pd.DataFrame({
    'Column': correlations.index,
    'Correlation_with_MathScore': correlations.values,
    'Abs_Correlation': np.abs(correlations.values)
})
correlation_df = correlation_df.sort_values('Abs_Correlation', ascending=False)

output_file = 'correlation_with_mathscore.csv'
correlation_df.to_csv(output_file, index=False)
print(f"\n✓ Full correlation table saved to: {output_file}")

# Display top 20 positive and negative correlations
print("\n" + "="*80)
print(" TOP 20 POSITIVE CORRELATIONS WITH MathScore")
print("="*80)
top_positive = correlations.head(20)
for i, (col, corr) in enumerate(top_positive.items(), 1):
    print(f"{i:2d}. {col:50s} {corr:+.4f}")

print("\n" + "="*80)
print(" TOP 20 NEGATIVE CORRELATIONS WITH MathScore")
print("="*80)
top_negative = correlations.tail(20)
for i, (col, corr) in enumerate(top_negative.items(), 1):
    print(f"{i:2d}. {col:50s} {corr:+.4f}")

# Visualization 1: Top correlations (both positive and negative)
print("\n📊 Generating visualization: Top 30 correlations...")

top_n = 15
top_pos = correlations.head(top_n)
top_neg = correlations.tail(top_n)
top_correlations = pd.concat([top_pos, top_neg])

fig, ax = plt.subplots(figsize=(12, 10))

colors = ['#00cc96' if x > 0 else '#ef553b' for x in top_correlations.values]
bars = ax.barh(range(len(top_correlations)), top_correlations.values, color=colors, alpha=0.7)

ax.set_yticks(range(len(top_correlations)))
ax.set_yticklabels(top_correlations.index, fontsize=9)
ax.set_xlabel('Pearson Correlation Coefficient with MathScore', fontsize=12, fontweight='bold')
ax.set_title('Top 15 Positive and Negative Correlations with Math Score\n(Potential Key Drivers)', 
             fontsize=13, fontweight='bold', pad=20)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(axis='x', alpha=0.3)

# Add values
for i, (bar, value) in enumerate(zip(bars, top_correlations.values)):
    x_pos = value + 0.01 if value > 0 else value - 0.01
    ha = 'left' if value > 0 else 'right'
    ax.text(x_pos, i, f'{value:.3f}', va='center', ha=ha, fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('correlation_top30_mathscore.png', dpi=300, bbox_inches='tight')
print("✓ Saved: correlation_top30_mathscore.png")
plt.show()

# Visualization 2: Distribution of correlation values
print("\n📊 Generating visualization: Distribution of all correlations...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Distribution of Correlation Coefficients with MathScore', fontsize=14, fontweight='bold')

# Histogram
axes[0].hist(correlations.values, bins=50, color='#636efa', alpha=0.7, edgecolor='black')
axes[0].axvline(correlations.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {correlations.mean():.4f}')
axes[0].axvline(correlations.median(), color='orange', linestyle=':', linewidth=2, 
                label=f'Median: {correlations.median():.4f}')
axes[0].set_xlabel('Correlation Coefficient', fontweight='bold')
axes[0].set_ylabel('Frequency', fontweight='bold')
axes[0].set_title('Histogram of Correlations', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Boxplot
bp = axes[1].boxplot([correlations.values], vert=True, patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('#636efa')
axes[1].set_ylabel('Correlation Coefficient', fontweight='bold')
axes[1].set_title('Boxplot of Correlations', fontweight='bold')
axes[1].set_xticklabels(['All Correlations'])
axes[1].grid(alpha=0.3)
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig('correlation_distribution_mathscore.png', dpi=300, bbox_inches='tight')
print("✓ Saved: correlation_distribution_mathscore.png")
plt.show()

# Summary statistics
print("\n" + "="*80)
print(" CORRELATION STATISTICS SUMMARY")
print("="*80)
print(f"Total numeric columns analyzed:     {len(correlations)}")
print(f"Mean correlation:                   {correlations.mean():+.4f}")
print(f"Median correlation:                 {correlations.median():+.4f}")
print(f"Standard deviation:                 {correlations.std():.4f}")
print(f"Minimum correlation:                {correlations.min():+.4f}")
print(f"Maximum correlation:                {correlations.max():+.4f}")
print(f"\nStrong positive correlations (>0.5): {(correlations > 0.5).sum()}")
print(f"Moderate positive correlations (0.3-0.5): {((correlations >= 0.3) & (correlations <= 0.5)).sum()}")
print(f"Weak correlations (-0.3 to 0.3):    {((correlations > -0.3) & (correlations < 0.3)).sum()}")
print(f"Moderate negative correlations (-0.5 to -0.3): {((correlations >= -0.5) & (correlations <= -0.3)).sum()}")
print(f"Strong negative correlations (<-0.5): {(correlations < -0.5).sum()}")

# Visualization 3: Missing data distribution by category
print("\n📊 Generating visualization: Missing data by category...")

# Calculate missing percentage for each column
missing_pct_train = (X_train.isnull().sum() / len(X_train) * 100).sort_values(ascending=False)
missing_pct_test = (X_test.isnull().sum() / len(X_test) * 100).sort_values(ascending=False)

# Define categories
categories = ['0-10%', '10-30%', '30-50%', '50-80%', '>80%']

# Count columns in each category for X_train
train_counts = [
    (missing_pct_train <= 10).sum(),
    ((missing_pct_train > 10) & (missing_pct_train <= 30)).sum(),
    ((missing_pct_train > 30) & (missing_pct_train <= 50)).sum(),
    ((missing_pct_train > 50) & (missing_pct_train <= 80)).sum(),
    (missing_pct_train > 80).sum()
]

# Count columns in each category for X_test
test_counts = [
    (missing_pct_test <= 10).sum(),
    ((missing_pct_test > 10) & (missing_pct_test <= 30)).sum(),
    ((missing_pct_test > 30) & (missing_pct_test <= 50)).sum(),
    ((missing_pct_test > 50) & (missing_pct_test <= 80)).sum(),
    (missing_pct_test > 80).sum()
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Distribution of Missing Data by Category', fontsize=16, fontweight='bold')

# X_train
x_pos = np.arange(len(categories))
bars1 = axes[0].bar(x_pos, train_counts, color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#8b0000'], alpha=0.7)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(categories)
axes[0].set_ylabel('Number of Columns', fontweight='bold')
axes[0].set_xlabel('Missing Data Percentage', fontweight='bold')
axes[0].set_title('X_train - Missing Data Categories', fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)
for i, (bar, count) in enumerate(zip(bars1, train_counts)):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(train_counts)*0.01, 
                str(count), ha='center', fontweight='bold', fontsize=10)

# X_test
bars2 = axes[1].bar(x_pos, test_counts, color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#8b0000'], alpha=0.7)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(categories)
axes[1].set_ylabel('Number of Columns', fontweight='bold')
axes[1].set_xlabel('Missing Data Percentage', fontweight='bold')
axes[1].set_title('X_test - Missing Data Categories', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)
for i, (bar, count) in enumerate(zip(bars2, test_counts)):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(test_counts)*0.01, 
                str(count), ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('missing_data_categories.png', dpi=300, bbox_inches='tight')
print("✓ Saved: missing_data_categories.png")
plt.show()

# Print detailed statistics
print("\n" + "="*80)
print(" MISSING DATA STATISTICS BY CATEGORY")
print("="*80)
print("\nX_train:")
for cat, count in zip(categories, train_counts):
    print(f"   {cat:10s} missing: {count:3d} columns ({count/len(X_train.columns)*100:5.1f}%)")

print("\nX_test:")
for cat, count in zip(categories, test_counts):
    print(f"   {cat:10s} missing: {count:3d} columns ({count/len(X_test.columns)*100:5.1f}%)")

print("\n" + "="*80)
print(" ✓ CORRELATION ANALYSIS COMPLETED")
print("="*80)
print("\n📁 Files generated:")
print("   • correlation_with_mathscore.csv (full table)")
print("   • correlation_top30_mathscore.png (top positive/negative)")
print("   • correlation_distribution_mathscore.png (distribution analysis)")
print("   • missing_data_categories.png (missing data by category)")
print()

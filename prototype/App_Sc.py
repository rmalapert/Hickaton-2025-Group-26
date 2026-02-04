import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# Configuration
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

print("="*80)
print(" ANALYSE EXPLORATOIRE DES DONNÉES PISA - PRÉDICTION DU SCORE EN MATHÉMATIQUES")
print("="*80)

# --- 1. Chargement des données ---

try:
    print("\n📂 Chargement des données...")
    X_train = pd.read_csv("X_train.csv", index_col=0) 
    X_test = pd.read_csv("X_test.csv", index_col=0)   
    y_train = pd.read_csv("y_train.csv", index_col=0)
    print("✓ Fichiers chargés avec succès.")
except FileNotFoundError:
    print("❌ Erreur: Assurez-vous que les fichiers CSV sont dans le même répertoire.")
    exit()

# Fusionner X_train et y_train pour l'analyse exploratoire
df_train = X_train.join(y_train, how='inner')

# Statistiques générales
print(f"\n📊 Dimensions des données:")
print(f"   • X_train: {X_train.shape[0]:,} observations × {X_train.shape[1]} variables")
print(f"   • X_test:  {X_test.shape[0]:,} observations × {X_test.shape[1]} variables")
print(f"   • y_train: {y_train.shape[0]:,} observations")
print(f"\n📈 Statistiques de la variable cible (MathScore):")
print(f"   • Moyenne:   {y_train['MathScore'].mean():.2f}")
print(f"   • Médiane:   {y_train['MathScore'].median():.2f}")
print(f"   • Écart-type: {y_train['MathScore'].std():.2f}")
print(f"   • Min:       {y_train['MathScore'].min():.2f}")
print(f"   • Max:       {y_train['MathScore'].max():.2f}")
zero_count = (y_train['MathScore'] == 0).sum()
print(f"   • Scores = 0: {zero_count:,} ({zero_count/len(y_train)*100:.1f}%)")

# --- 2. Fonctions d'Analyse et de Visualisation ---

def viz_missing_data(df, title):
    """Visualisation de la proportion de données manquantes (NaN) par colonne."""
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    # Calculer le pourcentage
    total_rows = len(df)
    missing_pct = (missing_data / total_rows) * 100
    
    if missing_data.empty:
        fig = go.Figure()
        fig.add_annotation(text="Aucune donnée manquante dans l'ensemble de données.", 
                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, 
                        font=dict(size=20, color="gray"))
        fig.update_layout(title=f'<b>{title}</b><br>Statut des Données Manquantes', height=600)
    else:
        fig = go.Figure(data=[
            go.Bar(x=missing_data.index, y=missing_data.values, name='Nombre de NaN', marker_color='#1f77b4'),
            go.Scatter(x=missing_data.index, y=missing_pct.values, name='% de NaN', yaxis='y2', 
                    mode='lines+markers', marker=dict(color="#0e4eff", size=8), line=dict(width=3))
        ])
        
        fig.update_layout(
            title=f'<b>Proportion de Données Manquantes (NaN) dans {title}</b>',
            xaxis_title='Variables',
            yaxis_title='Nombre d\'Observations Manquantes',
            yaxis2=dict(
                title='Pourcentage de Manquants (%)',
                overlaying='y',
                side='right',
                range=[0, missing_pct.max() * 1.1 if missing_pct.max() > 0 else 10]
            ),
            height=600,
            template="plotly_white",
            hovermode="x unified"
        )
    fig.show()

def viz_data_types(df):
    """Visualisation de la répartition des types de données (Numérique vs Catégorielle)."""
    # Déterminer les types
    dtypes = df.dtypes
    numerical_cols = dtypes[dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x) and x != object)].index.tolist()
    # Inclure 'object' (généralement strings) et autres non-numériques
    categorical_cols = dtypes[dtypes.apply(lambda x: pd.api.types.is_string_dtype(x) or x == object or x == 'category')].index.tolist()
    
    data_counts = {
        'Type': ['Numérique', 'Catégorielle'],
        'Nombre': [len(numerical_cols), len(categorical_cols)]
    }
    
    df_types = pd.DataFrame(data_counts)
    
    fig = px.pie(df_types, values='Nombre', names='Type', 
                title='<b>Répartition des Types de Variables</b>',
                color_discrete_sequence=['#2ca02c', '#d62728']) # Vert pour Numérique, Rouge pour Catégorielle
    
    fig.update_traces(textposition='inside', textinfo='percent+label', 
                marker=dict(line=dict(color='#FFFFFF', width=2)))
    fig.show()
    
def viz_unique_categorical_count(df):
    """Visualisation du nombre de modalités uniques pour les variables catégorielles."""
    
    # Identifier les colonnes catégorielles
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Calculer le nombre de valeurs uniques
    unique_counts = df[categorical_cols].nunique().sort_values(ascending=False)
    
    if unique_counts.empty:
        fig = go.Figure()
        fig.add_annotation(text="Aucune colonne catégorielle (type 'object' ou 'category') trouvée pour cette analyse.", 
                        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, 
                        font=dict(size=16, color="gray"))
        fig.update_layout(title='<b>Analyse des Modalités (Variables Catégorielles)</b>', height=600)
    else:
        # Filtrer pour ne pas afficher les colonnes avec trop de valeurs uniques (ex: identifiants qui sont mal typés)
        # Nous allons nous concentrer sur les 20 variables les plus pertinentes pour l'encodage
        max_to_show = 20 
        unique_counts = unique_counts[unique_counts < 1000].head(max_to_show) # Limite arbitraire
        
        fig = px.bar(unique_counts, 
                    y=unique_counts.index, 
                    x=unique_counts.values,
                    orientation='h',
                    title='<b>Nombre de Modalités Uniques par Variable Catégorielle</b><br>(Crucial pour l\'Encodage)',
                    color=unique_counts.values,
                    color_continuous_scale=px.colors.sequential.Viridis,
                    text=unique_counts.values)

        fig.update_layout(
            xaxis_title='Nombre de Modalités Uniques',
            yaxis_title='Variables Catégorielles',
            height=600,
            template="plotly_white",
            yaxis={'categoryorder':'total ascending'}
        )
        fig.show()
    
def viz_target_distribution(series):
    """Distribution de la variable cible (MathScore)."""
    fig = px.histogram(series, x=series.name, 
                    title='<b>Distribution de la Variable Cible (MathScore)</b>',
                    nbins=50, 
                    marginal="box", # Afficher le boxplot sur le côté pour les statistiques
                    color_discrete_sequence=['#9467bd']) # Couleur violette
    
    # Ajouter des lignes verticales pour la moyenne et la médiane
    mean_val = series.mean()
    median_val = series.median()
    
    fig.add_vline(x=mean_val, line_dash="dash", line_color="orange", annotation_text=f"Moyenne: {mean_val:.2f}")
    fig.add_vline(x=median_val, line_dash="dot", line_color="red", annotation_text=f"Médiane: {median_val:.2f}")
    
    fig.update_layout(xaxis_title='MathScore (Note/Performance)', height=600, template="plotly_white")
    fig.show()

def viz_outlier_detection(df, target_col):
    """Visualisation de détection d'outliers pour les variables numériques."""
    
    # Sélectionner quelques colonnes numériques (non-ID) pour l'exemple
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Exclure les colonnes qui ressemblent à des ID ou le score cible lui-même
    cols_to_plot = [col for col in numeric_cols if 'CNTSTUID' not in col and 'CNTRYID' not in col and col != target_col]
    
    if len(cols_to_plot) == 0:
        print("Peu de colonnes numériques pour cette visualisation.")
        return

    # Choisir les 8 premières colonnes pour l'affichage
    cols_to_plot = cols_to_plot[:8]

    fig = make_subplots(rows=2, cols=4, subplot_titles=[f'Boxplot de {col}' for col in cols_to_plot])
    
    row = 1
    col = 1
    for i, column in enumerate(cols_to_plot):
        fig.add_trace(
            go.Box(y=df[column], name=column, showlegend=False, boxpoints='outliers', 
                marker=dict(color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)])),
            row=row, col=col
        )
        col += 1
        if col > 4:
            col = 1
            row += 1
            
    fig.update_layout(title_text="<b>Détection d'Outliers dans les Variables Numériques Clés</b>", height=800, template="plotly_white")
    fig.show()
    
def viz_correlation_heatmap(df):
    """Matrice de corrélation interactive (Chaleur) avec la variable cible."""
    
    # Calcule la matrice de corrélation
    correlation_matrix = df.select_dtypes(include=np.number).corr()
    
    # Filtrer les colonnes qui ne sont pas pertinentes pour l'affichage (ID, etc.)
    cols_to_keep = [col for col in correlation_matrix.columns if 'ID' not in col]
    correlation_matrix = correlation_matrix.loc[cols_to_keep, cols_to_keep]

    # Créer le heatmap
    fig = px.imshow(correlation_matrix,
                    text_auto=".2f",
                    aspect="equal",
                    color_continuous_scale='RdBu_r', # Rouge-Bleu pour Corrélation
                    title='<b>Matrice de Corrélation des Variables Numériques</b>')

    fig.update_layout(height=800, width=1000, template="plotly_white")
    fig.show()

def viz_correlation_target(df, target_col='MathScore'):
    """Corrélation de chaque variable avec la cible (approche 'Key Driver' précoce)."""
    
    # Calculer la corrélation avec la cible et trier
    numeric_df = df.select_dtypes(include=np.number)
    correlations = numeric_df.corr()[target_col].drop(target_col).sort_values(ascending=False)
    
    # Prendre le top 15 positif et le top 15 négatif
    top_n = 10
    top_pos = correlations.head(top_n)
    top_neg = correlations.tail(top_n)
    
    # Combiner les deux listes pour un affichage équilibré
    viz_corrs = pd.concat([top_pos, top_neg])
    
    fig = go.Figure(data=[
        go.Bar(
            x=viz_corrs.index, 
            y=viz_corrs.values,
            marker_color=np.where(viz_corrs.values > 0, '#00cc96', '#ef553b'), # Vert pour positif, Rouge pour négatif
            text=[f'{val:.2f}' for val in viz_corrs.values],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='<b>Top 10 des Corrélations (Positives et Négatives) avec le MathScore</b>',
        xaxis_title='Variables',
        yaxis_title='Coefficient de Corrélation de Pearson',
        height=600,
        template="plotly_white"
    )
    fig.show()

def viz_target_analysis_detailed():
    """Analyse détaillée de la variable cible avec matplotlib."""
    print("\n📊 Génération: Analyse détaillée de la variable cible...")
    
    df_valid = df_train[df_train['MathScore'] > 0].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Analyse Complète de la Variable Cible (MathScore)', fontsize=16, fontweight='bold')
    
    # 1. Distribution (Histogram)
    axes[0, 0].hist(df_valid['MathScore'], bins=50, color='#5ab4ac', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(df_valid['MathScore'].mean(), color='red', linestyle='--', linewidth=2, label=f"Moyenne: {df_valid['MathScore'].mean():.1f}")
    axes[0, 0].axvline(df_valid['MathScore'].median(), color='orange', linestyle=':', linewidth=2, label=f"Médiane: {df_valid['MathScore'].median():.1f}")
    axes[0, 0].set_xlabel('Score en Mathématiques', fontweight='bold')
    axes[0, 0].set_ylabel('Fréquence', fontweight='bold')
    axes[0, 0].set_title('Distribution des Scores', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Boxplot
    bp = axes[0, 1].boxplot([df_valid['MathScore']], vert=True, patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#5ab4ac')
    axes[0, 1].set_ylabel('Score en Mathématiques', fontweight='bold')
    axes[0, 1].set_title('Détection des Outliers', fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Q-Q Plot (Normalité)
    stats.probplot(df_valid['MathScore'], dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Test de Normalité)', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Statistiques descriptives (texte)
    axes[1, 1].axis('off')
    stats_text = f"""
    STATISTIQUES DESCRIPTIVES
    ─────────────────────────
    
    Nombre d'observations: {len(df_valid):,}
    
    Moyenne:     {df_valid['MathScore'].mean():.2f}
    Médiane:     {df_valid['MathScore'].median():.2f}
    Écart-type:  {df_valid['MathScore'].std():.2f}
    
    Minimum:     {df_valid['MathScore'].min():.2f}
    Q1 (25%):    {df_valid['MathScore'].quantile(0.25):.2f}
    Q2 (50%):    {df_valid['MathScore'].quantile(0.50):.2f}
    Q3 (75%):    {df_valid['MathScore'].quantile(0.75):.2f}
    Maximum:     {df_valid['MathScore'].max():.2f}
    
    Skewness:    {df_valid['MathScore'].skew():.3f}
    Kurtosis:    {df_valid['MathScore'].kurtosis():.3f}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace', 
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('target_analysis_detailed.png', dpi=300, bbox_inches='tight')
    print("✓ Sauvegardé: target_analysis_detailed.png")
    plt.show()

def viz_feature_importance_correlation():
    """Visualisation des features les plus corrélées avec la cible (Key Drivers)."""
    print("\n📊 Génération: Top Features par corrélation...")
    
    df_valid = df_train[df_train['MathScore'] > 0].copy()
    numeric_df = df_valid.select_dtypes(include=np.number)
    
    # Calculer les corrélations
    correlations = numeric_df.corr()['MathScore'].drop('MathScore').sort_values(ascending=False)
    
    # Top 15 positives et négatives
    top_pos = correlations.head(15)
    top_neg = correlations.tail(15)
    top_features = pd.concat([top_pos, top_neg])
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = ['#00cc96' if x > 0 else '#ef553b' for x in top_features.values]
    bars = ax.barh(range(len(top_features)), top_features.values, color=colors, alpha=0.7)
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features.index, fontsize=9)
    ax.set_xlabel('Corrélation avec MathScore', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 des Corrélations Positives et Négatives avec le Score en Mathématiques\n(Key Drivers Potentiels)', 
                 fontsize=13, fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    # Ajouter les valeurs
    for i, (bar, value) in enumerate(zip(bars, top_features.values)):
        x_pos = value + 0.01 if value > 0 else value - 0.01
        ha = 'left' if value > 0 else 'right'
        ax.text(x_pos, i, f'{value:.3f}', va='center', ha=ha, fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('feature_importance_correlation.png', dpi=300, bbox_inches='tight')
    print("✓ Sauvegardé: feature_importance_correlation.png")
    plt.show()

def viz_data_quality_summary():
    """Résumé visuel de la qualité des données."""
    print("\n📊 Génération: Résumé de la qualité des données...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Résumé de la Qualité des Données', fontsize=16, fontweight='bold')
    
    # 1. Proportion de NaN par catégorie
    missing_train = (X_train.isnull().sum() / len(X_train) * 100)
    
    categories = ['0-10%', '10-30%', '30-50%', '>50%']
    train_counts = [
        (missing_train <= 10).sum(),
        ((missing_train > 10) & (missing_train <= 30)).sum(),
        ((missing_train > 30) & (missing_train <= 50)).sum(),
        (missing_train > 50).sum()
    ]
    
    axes[0].bar(categories, train_counts, color='#636efa', alpha=0.7)
    axes[0].set_ylabel('Nombre de Variables', fontweight='bold')
    axes[0].set_title('Distribution des Valeurs Manquantes\n(X_train)', fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(train_counts):
        axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')
    
    # 2. Types de données
    dtypes_count = X_train.dtypes.value_counts()
    axes[1].pie(dtypes_count.values, labels=[str(x) for x in dtypes_count.index], autopct='%1.1f%%', 
                colors=['#2ca02c', '#d62728', '#ff7f0e'], startangle=90)
    axes[1].set_title('Répartition des Types de Données', fontweight='bold')
    
    # 3. Distribution des scores (avec/sans zéros)
    all_scores = df_train['MathScore']
    valid_scores = df_train[df_train['MathScore'] > 0]['MathScore']
    
    data_to_plot = [valid_scores, all_scores]
    bp = axes[2].boxplot(data_to_plot, labels=['Scores > 0', 'Tous scores'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#5ab4ac', '#9467bd']):
        patch.set_facecolor(color)
    axes[2].set_ylabel('Score en Mathématiques', fontweight='bold')
    axes[2].set_title('Comparaison des Distributions', fontweight='bold')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_quality_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Sauvegardé: data_quality_summary.png")
    plt.show()

# --- 3. Exécution des Visualisations ---

print("\n" + "="*80)
print(" SECTION 1: ANALYSE DE LA VARIABLE CIBLE")
print("="*80)

viz_target_analysis_detailed()

print("\n" + "="*80)
print(" SECTION 2: IDENTIFICATION DES KEY DRIVERS")
print("="*80)

viz_feature_importance_correlation()

print("\n" + "="*80)
print(" SECTION 3: QUALITÉ ET STRUCTURE DES DONNÉES")
print("="*80)

viz_data_quality_summary()

print("\n" + "="*80)
print(" SECTION 4: VISUALISATIONS INTERACTIVES (PLOTLY)")
print("="*80)

print("\n📊 VIZ 1 : Proportion des NaN")
viz_missing_data(df_train, "X_train + y_train")
viz_missing_data(X_test, "X_test")

print("\n📊 VIZ 2 : Répartition des types de données")
viz_data_types(X_train)

print("\n📊 VIZ 3 : Analyse des modalités catégorielles")
viz_unique_categorical_count(X_train)

print("\n📊 VIZ 4 : Distribution de la variable cible (Plotly)")
viz_target_distribution(df_train['MathScore'])

print("\n📊 VIZ 5 : Détection d'Outliers (Boxplots multiples)")
viz_outlier_detection(df_train, 'MathScore')

print("\n📊 VIZ 6 : Heatmap de Corrélation")
viz_correlation_heatmap(df_train)

print("\n📊 VIZ 7 : Corrélation avec la cible (Plotly)")
viz_correlation_target(df_train, 'MathScore')

# --- 4. Section Modeling & Explainability (Templates) ---

print("\n" + "="*80)
print(" SECTION 5: TEMPLATES POUR MODELING & EXPLAINABILITY")
print("="*80)
print("\n⚠️  Ces visualisations sont des templates à remplir après l'entraînement de vos modèles.")

def viz_model_comparison():
    """Template: Comparaison des performances des modèles."""
    print("\n📊 Template: Comparaison des modèles...")
    
    # REMPLACER PAR VOS VRAIES MÉTRIQUES
    models = ['Baseline\n(Moyenne)', 'Régression\nLinéaire', 'Random\nForest', 'Gradient\nBoosting', 'XGBoost']
    rmse_values = [122.18, 105.5, 95.2, 88.7, 85.3]  # Exemple
    mae_values = [98.5, 85.3, 78.1, 72.4, 69.8]      # Exemple
    r2_values = [0.0, 0.35, 0.52, 0.61, 0.65]        # Exemple
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Comparaison des Performances des Modèles', fontsize=16, fontweight='bold')
    
    # RMSE
    bars1 = axes[0].bar(models, rmse_values, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd'], alpha=0.7)
    axes[0].set_ylabel('RMSE', fontweight='bold')
    axes[0].set_title('Root Mean Square Error\n(plus bas = meilleur)', fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars1, rmse_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.1f}', 
                    ha='center', fontweight='bold')
    
    # MAE
    bars2 = axes[1].bar(models, mae_values, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd'], alpha=0.7)
    axes[1].set_ylabel('MAE', fontweight='bold')
    axes[1].set_title('Mean Absolute Error\n(plus bas = meilleur)', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars2, mae_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.1f}', 
                    ha='center', fontweight='bold')
    
    # R²
    bars3 = axes[2].bar(models, r2_values, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd'], alpha=0.7)
    axes[2].set_ylabel('R² Score', fontweight='bold')
    axes[2].set_title('Coefficient de Détermination\n(plus haut = meilleur)', fontweight='bold')
    axes[2].set_ylim(0, 1)
    axes[2].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars3, r2_values):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', 
                    ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Sauvegardé: model_comparison.png")
    plt.show()

def viz_feature_importance_shap_template():
    """Template: Feature importance style SHAP."""
    print("\n📊 Template: Feature Importance (SHAP-style)...")
    
    # REMPLACER PAR VOS VRAIES VALEURS SHAP
    features = [
        'reading_q*_average_score',
        'math_q*_average_score',
        'ST255 (Motivation)',
        'AGE',
        'ST331 (Temps lecture)',
        'WB153 (Bien-être)',
        'GRADE',
        'ST350 (Confiance)',
        'IC176 (Ressources TIC)',
        'MATHEASE (Facilité maths)'
    ]
    importance = [12.5, 10.8, 8.3, 7.2, 6.5, 5.9, 5.1, 4.7, 4.2, 3.8]  # Exemple
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
    bars = ax.barh(range(len(features)), importance, color=colors, alpha=0.8)
    
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=11)
    ax.set_xlabel('Importance Moyenne (SHAP Value)', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 des Features les Plus Importantes\n(Analyse d\'Explainabilité - SHAP)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Ajouter les valeurs
    for i, (bar, value) in enumerate(zip(bars, importance)):
        ax.text(value + 0.2, i, f'{value:.1f}', va='center', fontweight='bold')
    
    # Note explicative
    note = "Note: Les valeurs SHAP mesurent l'impact moyen de chaque variable\nsur les prédictions du modèle (en unités de score)."
    ax.text(0.98, 0.02, note, transform=ax.transAxes, fontsize=9, 
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('feature_importance_shap.png', dpi=300, bbox_inches='tight')
    print("✓ Sauvegardé: feature_importance_shap.png")
    plt.show()

def viz_residuals_analysis_template():
    """Template: Analyse des résidus du modèle."""
    print("\n📊 Template: Analyse des résidus...")
    
    # REMPLACER PAR VOS VRAIES PRÉDICTIONS
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.normal(160, 80, n_samples)
    y_pred = y_true + np.random.normal(0, 20, n_samples)
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Analyse des Résidus du Modèle (Diagnostic)', fontsize=16, fontweight='bold')
    
    # 1. Résidus vs Prédictions
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=10, c='#1f77b4')
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Valeurs Prédites', fontweight='bold')
    axes[0, 0].set_ylabel('Résidus', fontweight='bold')
    axes[0, 0].set_title('Résidus vs Prédictions', fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Distribution des résidus
    axes[0, 1].hist(residuals, bins=50, color='#5ab4ac', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(residuals.mean(), color='red', linestyle='--', linewidth=2, label=f'Moyenne: {residuals.mean():.2f}')
    axes[0, 1].set_xlabel('Résidus', fontweight='bold')
    axes[0, 1].set_ylabel('Fréquence', fontweight='bold')
    axes[0, 1].set_title('Distribution des Résidus', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot des Résidus', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Prédictions vs Valeurs Réelles
    axes[1, 1].scatter(y_true, y_pred, alpha=0.5, s=10, c='#ff7f0e')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[1, 1].plot(lims, lims, 'r--', linewidth=2, label='Prédiction Parfaite')
    axes[1, 1].set_xlabel('Valeurs Réelles', fontweight='bold')
    axes[1, 1].set_ylabel('Valeurs Prédites', fontweight='bold')
    axes[1, 1].set_title('Prédictions vs Valeurs Réelles', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('residuals_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Sauvegardé: residuals_analysis.png")
    plt.show()

# Générer les templates
viz_model_comparison()
viz_feature_importance_shap_template()
viz_residuals_analysis_template()

print("\n" + "="*80)
print(" ✓ ANALYSE TERMINÉE - TOUS LES GRAPHIQUES ONT ÉTÉ GÉNÉRÉS")
print("="*80)
print("\n📁 Fichiers générés:")
print("   • target_analysis_detailed.png")
print("   • feature_importance_correlation.png")
print("   • data_quality_summary.png")
print("   • model_comparison.png (template)")
print("   • feature_importance_shap.png (template)")
print("   • residuals_analysis.png (template)")
print("\n💡 Note: Les graphiques 'template' contiennent des données d'exemple.")
print("   Remplacez-les par vos vraies métriques après l'entraînement de vos modèles.\n")
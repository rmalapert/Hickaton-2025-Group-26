# -----------------------------------------------------------------------------
# ARTICLE 10 – PROPRIETE INTELLECTUELLE & LICENSE HEADER
# -----------------------------------------------------------------------------
# The present code is developed during a contest called « HI!CKATHON » organized
# at the request of HEC Paris and Institut Polytechnique de Paris, in the
# context of the Hi ! PARIS Center.
#
# License: LGPL v3 (Open Source)
# Authors: Gabin BIGARET, Antoine LOUVET, Rémi MALAPERT, Othmane NAMMOUS,
#          Valentin SENAUX et Tharushan UTHAYAKUMAR 
# -----------------------------------------------------------------------------

"""
📊 Analyse Exploratoire des Données PISA - HI!CKATHON 2025

Objectif: Analyser en profondeur la structure, la qualité et les caractéristiques 
des données PISA pour préparer la modélisation prédictive de la réussite scolaire.
"""

# =============================================================================
# 1️⃣ IMPORT DES BIBLIOTHÈQUES
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

# Configuration de l'affichage pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)

# Style des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ Bibliothèques importées avec succès!")

# =============================================================================
# 2️⃣ CHARGEMENT ET INSPECTION INITIALE DES DONNÉES
# =============================================================================

print("\n" + "="*80)
print("📂 CHARGEMENT DES DONNÉES")
print("="*80)

# Chargement des données X_train
print("\n📂 Chargement de X_train.csv...")
try:
    X_train = pd.read_csv('X_train.csv', low_memory=False)
    print(f"✅ X_train chargé: {X_train.shape[0]} lignes × {X_train.shape[1]} colonnes")
except Exception as e:
    print(f"❌ Erreur lors du chargement de X_train: {e}")
    X_train = None

# Chargement des données y_train
print("\n📂 Chargement de y_train.csv...")
try:
    y_train = pd.read_csv('y_train.csv', low_memory=False)
    print(f"✅ y_train chargé: {y_train.shape[0]} lignes × {y_train.shape[1]} colonnes")
except Exception as e:
    print(f"❌ Erreur lors du chargement de y_train: {e}")
    y_train = None

# Affichage des premières lignes
if X_train is not None:
    print("\n" + "="*80)
    print("📋 Aperçu des premières lignes de X_train:")
    print("="*80)
    print(X_train.head())

# =============================================================================
# 3️⃣ INFORMATIONS GÉNÉRALES
# =============================================================================

if X_train is not None:
    print("\n" + "="*80)
    print("📊 INFORMATIONS GÉNÉRALES SUR X_TRAIN")
    print("="*80)
    print(f"\n📐 Dimensions: {X_train.shape[0]:,} lignes × {X_train.shape[1]:,} colonnes")
    print(f"💾 Utilisation mémoire: {X_train.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\n🔢 Types de données:")
    print(X_train.dtypes.value_counts())
    
    print("\n" + "="*80)
    print("📈 STATISTIQUES DE BASE")
    print("="*80)
    X_train.info()

# =============================================================================
# 4️⃣ ANALYSE DÉTAILLÉE DES COLONNES ET TAUX DE REMPLISSAGE
# =============================================================================

if X_train is not None:
    print("\n" + "="*80)
    print("📊 ANALYSE DÉTAILLÉE DES COLONNES")
    print("="*80)
    
    column_info = pd.DataFrame({
        'Colonne': X_train.columns,
        'Type': X_train.dtypes,
        'Valeurs_Non_Nulles': X_train.count(),
        'Valeurs_Nulles': X_train.isnull().sum(),
        'Pct_Manquant': (X_train.isnull().sum() / len(X_train) * 100).round(2),
        'Pct_Remplissage': ((1 - X_train.isnull().sum() / len(X_train)) * 100).round(2),
        'Valeurs_Uniques': X_train.nunique(),
    })
    
    column_info = column_info.sort_values('Pct_Manquant', ascending=False).reset_index(drop=True)
    
    print("\n📊 TABLEAU RÉCAPITULATIF DES COLONNES (Top 30):")
    print(column_info.head(30).to_string())
    
    # Statistiques globales
    print("\n" + "="*80)
    print("📉 STATISTIQUES GLOBALES SUR LES VALEURS MANQUANTES")
    print("="*80)
    print(f"✓ Colonnes totales: {len(column_info)}")
    print(f"✓ Colonnes complètes (0% manquant): {(column_info['Pct_Manquant'] == 0).sum()}")
    print(f"✓ Colonnes avec <10% manquant: {(column_info['Pct_Manquant'] < 10).sum()}")
    print(f"✓ Colonnes avec 10-50% manquant: {((column_info['Pct_Manquant'] >= 10) & (column_info['Pct_Manquant'] < 50)).sum()}")
    print(f"⚠ Colonnes avec >50% manquant: {(column_info['Pct_Manquant'] >= 50).sum()}")
    print(f"🔴 Colonnes entièrement vides (100% manquant): {(column_info['Pct_Manquant'] == 100).sum()}")

# =============================================================================
# 5️⃣ VISUALISATION DES VALEURS MANQUANTES
# =============================================================================

if X_train is not None:
    print("\n" + "="*80)
    print("📊 GÉNÉRATION DES GRAPHIQUES...")
    print("="*80)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    top_missing = column_info[column_info['Pct_Manquant'] > 0].head(30)
    
    if len(top_missing) > 0:
        axes[0].barh(range(len(top_missing)), top_missing['Pct_Manquant'], 
                     color=plt.cm.RdYlGn_r(top_missing['Pct_Manquant']/100))
        axes[0].set_yticks(range(len(top_missing)))
        axes[0].set_yticklabels(top_missing['Colonne'], fontsize=8)
        axes[0].set_xlabel('Pourcentage de Valeurs Manquantes (%)', fontsize=11)
        axes[0].set_title('🔴 Top 30 Colonnes avec le Plus de Valeurs Manquantes', fontsize=13, fontweight='bold')
        axes[0].axvline(x=50, color='red', linestyle='--', linewidth=2, label='Seuil critique (50%)')
        axes[0].legend()
        axes[0].grid(axis='x', alpha=0.3)
    
    axes[1].hist(column_info['Pct_Manquant'], bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=50, color='red', linestyle='--', linewidth=2, label='Seuil critique (50%)')
    axes[1].set_xlabel('Pourcentage de Valeurs Manquantes (%)', fontsize=11)
    axes[1].set_ylabel('Nombre de Colonnes', fontsize=11)
    axes[1].set_title('📊 Distribution des Taux de Valeurs Manquantes', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('valeurs_manquantes.png', dpi=150, bbox_inches='tight')
    print("✅ Graphique sauvegardé: valeurs_manquantes.png")
    plt.close()

# =============================================================================
# 6️⃣ ANALYSE DES TYPES DE DONNÉES
# =============================================================================

if X_train is not None:
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category', 'bool']).columns
    datetime_cols = X_train.select_dtypes(include=['datetime64']).columns
    
    print("\n" + "="*80)
    print("📊 DISTRIBUTION DES TYPES DE DONNÉES")
    print("="*80)
    print(f"✓ Variables numériques: {len(numeric_cols)}")
    print(f"✓ Variables catégorielles: {len(categorical_cols)}")
    print(f"✓ Variables date/temps: {len(datetime_cols)}")

# =============================================================================
# 7️⃣ DÉTECTION DES CODES MANQUANTS SPÉCIFIQUES PISA
# =============================================================================

if X_train is not None:
    print("\n" + "="*80)
    print("🔍 DÉTECTION DES CODES MANQUANTS PISA")
    print("="*80)
    
    missing_codes = [9999, 999, 99, 99999, 'r', 'n', 'a']
    pisa_missing_summary = []
    
    for col in X_train.columns:
        code_counts = {}
        total_codes = 0
        
        for code in missing_codes:
            try:
                count = (X_train[col] == code).sum()
                if count > 0:
                    code_counts[code] = count
                    total_codes += count
            except:
                pass
        
        if total_codes > 0:
            pisa_missing_summary.append({
                'Colonne': col,
                'Total_Codes_PISA': total_codes,
                'Pct_Total': (total_codes / len(X_train) * 100).round(2),
            })
    
    if len(pisa_missing_summary) > 0:
        pisa_missing_df = pd.DataFrame(pisa_missing_summary).sort_values('Total_Codes_PISA', ascending=False)
        print(f"\n✓ {len(pisa_missing_df)} colonnes contiennent des codes manquants PISA")
        print(f"✓ Total de codes manquants détectés: {pisa_missing_df['Total_Codes_PISA'].sum():,}")
        print("\nTop 20:")
        print(pisa_missing_df.head(20).to_string())
    else:
        print("✅ Aucun code manquant PISA détecté")

# =============================================================================
# 8️⃣ RAPPORT DE QUALITÉ ET RECOMMANDATIONS
# =============================================================================

if X_train is not None:
    print("\n" + "="*80)
    print("📋 RAPPORT DE QUALITÉ DES DONNÉES ET RECOMMANDATIONS")
    print("="*80)
    
    cols_to_drop = column_info[column_info['Pct_Manquant'] > 50]['Colonne'].tolist()
    cols_to_impute = column_info[(column_info['Pct_Manquant'] >= 10) & 
                                  (column_info['Pct_Manquant'] <= 50)]['Colonne'].tolist()
    good_quality_cols = column_info[column_info['Pct_Manquant'] < 10]['Colonne'].tolist()
    
    print(f"\n🔴 COLONNES À SUPPRIMER (>50% manquant): {len(cols_to_drop)}")
    print(f"⚠️  COLONNES À IMPUTER (10-50% manquant): {len(cols_to_impute)}")
    print(f"✅ COLONNES DE BONNE QUALITÉ (<10% manquant): {len(good_quality_cols)}")
    
    print("\n" + "="*80)
    print("💡 RÉSUMÉ DES RECOMMANDATIONS")
    print("="*80)
    print(f"""
    1️⃣ Supprimer {len(cols_to_drop)} colonnes avec >50% de valeurs manquantes
    2️⃣ Impluter {len(cols_to_impute)} colonnes avec 10-50% de valeurs manquantes
    3️⃣ Remplacer les codes PISA (9999, 999, 99, 'r', 'n') par NaN
    4️⃣ Standardiser les {len(numeric_cols)} variables numériques
    5️⃣ Encoder les {len(categorical_cols)} variables catégorielles
    """)

# =============================================================================
# 9️⃣ EXPORT DES RÉSULTATS
# =============================================================================

if X_train is not None:
    try:
        column_info.to_csv('rapport_qualite_colonnes.csv', index=False, encoding='utf-8')
        print("\n✅ Rapport exporté vers: rapport_qualite_colonnes.csv")
        
        synthese = pd.DataFrame({
            'Métrique': [
                'Total Colonnes',
                'Total Lignes',
                'Variables Numériques',
                'Variables Catégorielles',
                'Colonnes Complètes',
                'Colonnes à Supprimer',
                'Colonnes à Imputer',
                'Taux Moyen Remplissage (%)',
            ],
            'Valeur': [
                len(X_train.columns),
                len(X_train),
                len(numeric_cols),
                len(categorical_cols),
                (column_info['Pct_Manquant'] == 0).sum(),
                len(cols_to_drop),
                len(cols_to_impute),
                column_info['Pct_Remplissage'].mean().round(2),
            ]
        })
        
        synthese.to_csv('rapport_synthese.csv', index=False, encoding='utf-8')
        print("✅ Synthèse exportée vers: rapport_synthese.csv")
        
        print("\n📊 Rapport de synthèse:")
        print(synthese.to_string(index=False))
        
    except Exception as e:
        print(f"❌ Erreur lors de l'export: {e}")

print("\n" + "="*80)
print("🎓 ANALYSE TERMINÉE AVEC SUCCÈS!")
print("="*80)
print("\n📁 Fichiers générés:")
print("  - rapport_qualite_colonnes.csv")
print("  - rapport_synthese.csv")
print("  - valeurs_manquantes.png")

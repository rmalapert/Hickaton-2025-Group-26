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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings

# --- Preprocessing & Pipeline ---
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Models (Tree-based are best for PISA) ---
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

# --- Metrics ---
from sklearn.metrics import mean_squared_error, r2_score, f1_score, accuracy_score, classification_report

# --- Explainability (Critical for Education) ---
import shap

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

# =============================================================================
# 1. CONFIGURATION & CHARGEMENT
# =============================================================================

# TODO: Ajuster ces variables dès ouverture des données
FILE_PATH = "X_train.csv"  
TARGET_COL = "PV1MATH"          # Ex: Score de Maths (Souvent PV1MATH, PV1READ...)
SENSITIVE_COL = "ST004D01T"     # Ex: Code PISA pour le Genre (Homme/Femme) - Pour l'éthique
ID_COL = "CNTRYID"              # Ou autre identifiant élève/école
IS_REGRESSION = True            # True si on prédit un score (0-1000), False si on prédit "Décrochage" (0/1)

def load_pisa_data(path):
    print(f"--- Chargement des données PISA : {path} ---")
    # low_memory=False est souvent nécessaire pour PISA car les fichiers sont gros
    try:
        df = pd.read_csv(path, low_memory=False)
    except:
        # Fallback si séparateur différent
        df = pd.read_csv(path, sep=';', low_memory=False)
    
    print(f"Dimensions initiales: {df.shape}")
    return df

# =============================================================================
# 2. NETTOYAGE SPECIFIQUE PISA (Preprocessing I)
# =============================================================================

def clean_pisa_features(df):
    """
    Nettoyage adapté aux sondages de l'OCDE.
    """
    df_clean = df.copy()
    
    # 1. Gestion des codes "Manquant" spécifiques à PISA
    # Souvent, 9999, 999, 99, 'r', 'n' signifient "Pas de réponse"
    missing_codes = [9999, 999, 99, 99999, 'r', 'n', 'a'] 
    df_clean = df_clean.replace(missing_codes, np.nan)
    
    # 2. Conversion des colonnes "Object" qui devraient être numériques
    # (PISA met parfois des strings dans des colonnes de scores)
    for col in df_clean.select_dtypes(include='object').columns:
        try:
            df_clean[col] = pd.to_numeric(df_clean[col])
        except:
            pass # Si ça rate, c'est que c'est vraiment du texte

    # 3. Suppression des colonnes avec trop de vides (>50%)
    # Dans PISA, certaines questions ne sont posées qu'à 10% des élèves (Matrix Sampling)
    threshold = 0.5
    null_counts = df_clean.isnull().mean()
    to_drop = null_counts[null_counts > threshold].index
    print(f"Colonnes supprimées car trop vides (>50%): {len(to_drop)}")
    df_clean = df_clean.drop(columns=to_drop)
    
    return df_clean

# =============================================================================
# 3. PIPELINE ROBUSTE (Preprocessing II)
# =============================================================================

def build_pipeline(X_train):
    """
    Construit un pipeline qui gère automatiquement les types de données.
    Pour PISA, on sépare Numérique et Catégoriel.
    """
    # Détection automatique des types
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'category', 'bool']).columns
    
    print(f"\n--- Construction du Pipeline ---")
    print(f"Variables Numériques detectées: {len(numeric_features)}")
    print(f"Variables Catégorielles detectées: {len(categorical_features)}")

    # 1. Pipeline Numérique
    # Pour PISA, la médiane est plus robuste aux outliers (élèves très brillants ou très faibles)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())
    ])

    # 2. Pipeline Catégoriel
    # Pour les questions QCM, "Most Frequent" est une bonne approximation
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 3. Assemblage
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        verbose_feature_names_out=False
    )
    
    return preprocessor

# =============================================================================
# 4. ENTRAINEMENT & EVALUATION (Machine Learning)
# =============================================================================

def train_and_evaluate(X_train, y_train, X_test, y_test, preprocessor, is_regression=True):
    
    # Choix du modèle : XGBoost est ROI sur les données tabulaires PISA
    if is_regression:
        model = XGBRegressor(
            n_estimators=200, 
            learning_rate=0.05, 
            max_depth=6, 
            random_state=42, 
            n_jobs=-1
        )
        metric_name = "RMSE"
    else:
        model = XGBClassifier(
            n_estimators=200, 
            learning_rate=0.05, 
            max_depth=6, 
            random_state=42, 
            n_jobs=-1
        )
        metric_name = "F1-Score"

    # Création du Pipeline complet
    full_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    print(f"\n--- Démarrage de l'entraînement ({'Regression' if is_regression else 'Classification'}) ---")
    start = time.time()
    full_pipe.fit(X_train, y_train)
    end = time.time()
    print(f"Temps d'entraînement: {end - start:.2f} sec (Impact Environnemental)")

    # Prédictions
    y_pred = full_pipe.predict(X_test)

    # Métriques
    print("\n--- Résultats ---")
    if is_regression:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")
    else:
        print(classification_report(y_test, y_pred))

    return full_pipe, y_pred

# =============================================================================
# 5. FAIRNESS CHECK (Le "Truc en plus" pour l'Éducation)
# =============================================================================

def check_fairness(df_test, y_test, y_pred, sensitive_col, is_regression=True):
    """
    Vérifie si le modèle performe aussi bien pour le groupe A (ex: Filles) 
    que pour le groupe B (ex: Garçons).
    """
    if sensitive_col not in df_test.columns:
        print(f"Colonne sensible '{sensitive_col}' non trouvée pour le test d'équité.")
        return

    print(f"\n--- Fairness Check (Équité) sur '{sensitive_col}' ---")
    
    # Création d'un DF temporaire pour l'analyse
    analysis_df = pd.DataFrame({
        'Real': y_test,
        'Pred': y_pred,
        'Group': df_test[sensitive_col]
    })
    
    # Calcul de l'erreur par groupe
    if is_regression:
        analysis_df['Error'] = (analysis_df['Real'] - analysis_df['Pred'])**2 # MSE
        grouped = analysis_df.groupby('Group')['Error'].mean().apply(np.sqrt) # RMSE
        print("RMSE par groupe :")
        print(grouped)
        
        # Graphique
        plt.figure(figsize=(6, 4))
        sns.barplot(x=grouped.index, y=grouped.values, palette="viridis")
        plt.title(f"Erreur du modèle selon {sensitive_col}")
        plt.ylabel("RMSE (Plus bas est mieux)")
        plt.show()
    else:
        # Pour la classification, on regarde le taux de précision par groupe
        analysis_df['Correct'] = (analysis_df['Real'] == analysis_df['Pred'])
        grouped = analysis_df.groupby('Group')['Correct'].mean()
        print("Accuracy par groupe :")
        print(grouped)

# =============================================================================
# 6. EXPLICABILITÉ (SHAP)
# =============================================================================

def run_shap_analysis(pipeline, X_train, X_test):
    print("\n--- Analyse d'Explicabilité (SHAP) ---")
    
    # 1. Extraction
    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocessor']
    
    # 2. Transformation (SHAP a besoin des données pré-traitées)
    # On prend un échantillon car PISA est trop gros pour SHAP complet
    sample_size = 200 
    X_sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=42)
    X_transformed = preprocessor.transform(X_sample)
    
    # Récupération des noms de features (Astuce pour OneHotEncoder)
    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        feature_names = [f"Feature_{i}" for i in range(X_transformed.shape[1])]
    
    # 3. Calcul SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)
    
    # 4. Plot Global
    plt.figure()
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
    plt.title("Facteurs influençant la réussite (Global)")
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    # 1. Load Data (Simulation ici - DECOMMENTER LA LIGNE SUIVANTE LE JOUR J)
    # df = load_pisa_data(FILE_PATH_TRAIN)
    
    # --- MOCK DATA POUR TESTER LE CODE MAINTENANT ---
    df = pd.DataFrame({
        'ST004D01T': np.random.choice(['Female', 'Male'], 1000), # Genre
        'ESCS': np.random.normal(0, 1, 1000),                   # Statut socio-éco
        'TIME_STUDY': np.random.randint(0, 60, 1000),           # Temps d'étude
        'JOY_READ': np.random.choice(['Agree', 'Disagree', np.nan], 1000), # Plaisir lecture
        'PV1MATH': np.random.normal(500, 100, 1000)             # Target Score
    })
    TARGET_COL = 'PV1MATH'
    SENSITIVE_COL = 'ST004D01T'
    # ------------------------------------------------
    
    # 2. Nettoyage
    df_clean = clean_pisa_features(df)
    
    # 3. Split X/y
    if TARGET_COL in df_clean.columns:
        y = df_clean[TARGET_COL]
        X = df_clean.drop(columns=[TARGET_COL])
        
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 4. Build Pipeline
        preprocessor = build_pipeline(X_train)
        
        # 5. Train
        pipeline, y_pred = train_and_evaluate(X_train, y_train, X_test, y_test, preprocessor, is_regression=IS_REGRESSION)
        
        # 6. Fairness Check (Le point fort pour le jury)
        # On vérifie si notre IA est "juste" entre les genres
        check_fairness(X_test, y_test, y_pred, SENSITIVE_COL, is_regression=IS_REGRESSION)
        
        # 7. Explainability
        # Pour comprendre POURQUOI un élève réussit ou échoue
        try:
            run_shap_analysis(pipeline, X_train, X_test)
        except Exception as e:
            print(f"SHAP Error: {e}")
            
    else:
        print(f"Erreur: Colonne target '{TARGET_COL}' introuvable.")
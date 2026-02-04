import pandas as pd
import os

def create_subset(file_path, output_dir, n_rows=10000):
    """
    Lit les n_rows premières lignes d'un fichier CSV et crée un subset.
    Le fichier de sortie sera sauvegardé dans output_dir.
    """
    if not os.path.exists(file_path):
        print(f"Fichier non trouvé : {file_path}")
        return

    print(f"Lecture des {n_rows} premières lignes de {file_path}...")
    try:
        # Modification pour éviter MemoryError : on lit seulement les n_rows premières lignes
        # Cela évite de charger tout le fichier en mémoire
        df_subset = pd.read_csv(file_path, nrows=n_rows, low_memory=False)
        
        # Générer le nom de fichier de sortie
        base_name = os.path.basename(file_path)
        name, ext = os.path.splitext(base_name)
        output_file = os.path.join(output_dir, f"{name}_subset{ext}")
        
        print(f"  Sauvegarde dans {output_file}...")
        df_subset.to_csv(output_file, index=False)
        print("  Terminé.")
        
    except Exception as e:
        print(f"Erreur lors du traitement de {file_path} : {e}")

def main():
    files_to_process = ['X_train.csv', 'y_train.csv', 'X_test.csv']
    output_dir = 'subset'
    
    # Création du dossier subset s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Dossier '{output_dir}' créé.")
    
    for file in files_to_process:
        create_subset(file, output_dir, n_rows=10000)

if __name__ == "__main__":
    main()

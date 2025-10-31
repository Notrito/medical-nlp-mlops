"""
Script de preparación de datos para Medical NLP MLOps

Proceso:
1. Carga datos raw
2. Limpia valores nulos
3. Crea categoría specialty_clean (Top N + Others)
4. Aplica chunking a textos largos
5. Train/test split estratificado
6. Guarda datasets procesados
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys
sys.path.append('.')
import config


def create_chunks(text, chunk_size=2000, overlap=200):
    """
    Divide un texto en chunks con overlap.
    
    Args:
        text: Texto a dividir
        chunk_size: Tamaño de cada chunk en caracteres
        overlap: Solapamiento entre chunks
    
    Returns:
        Lista de chunks
    """
    if pd.isna(text):
        return []
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        if end >= len(text):
            break
            
        start = end - overlap
    
    return chunks


def create_specialty_mapping(df, top_n=19):
    """
    Crea mapeo de especialidades: Top N + Others
    
    Args:
        df: DataFrame con columna medical_specialty
        top_n: Número de especialidades top a mantener
    
    Returns:
        Lista de especialidades top
    """
    top_specialties = df[config.LABEL_COLUMN].value_counts().head(top_n).index.tolist()
    return top_specialties


def process_dataset_with_chunks(df, top_specialties, chunk_size=2000, overlap=200):
    """
    Procesa el dataset aplicando chunking donde sea necesario.
    
    Args:
        df: DataFrame original
        top_specialties: Lista de especialidades top
        chunk_size: Tamaño de chunks
        overlap: Solapamiento entre chunks
    
    Returns:
        DataFrame procesado con chunks
    """
    processed_data = []
    
    for idx, row in df.iterrows():
        text = row[config.TEXT_COLUMN]
        specialty = row[config.LABEL_COLUMN]
        
        # Mapear a specialty_clean (Top N + Others)
        specialty_clean = specialty if specialty in top_specialties else 'Others'
        
        # Crear chunks
        chunks = create_chunks(text, chunk_size, overlap)
        
        # Añadir cada chunk como una nueva fila
        for chunk_idx, chunk in enumerate(chunks):
            processed_data.append({
                config.TEXT_COLUMN: chunk,
                config.CLEAN_LABEL_COLUMN: specialty_clean,
                'original_id': idx,
                'chunk_id': chunk_idx,
                'total_chunks': len(chunks)
            })
    
    return pd.DataFrame(processed_data)


def main():
    """
    Función principal que orquesta todo el proceso de preparación de datos.
    """
    print("=" * 60)
    print("PREPARACIÓN DE DATOS - Medical NLP MLOps")
    print("=" * 60)
    
    # 1. Cargar datos raw
    print(f"\n[1/6] Cargando datos desde {config.RAW_DATA_PATH}...")
    df = pd.read_csv(config.RAW_DATA_PATH)
    print(f"   ✓ Datos cargados: {len(df)} muestras")
    
    # 2. Limpiar valores nulos en transcription
    print(f"\n[2/6] Limpiando valores nulos...")
    initial_count = len(df)
    df = df.dropna(subset=[config.TEXT_COLUMN])
    removed_count = initial_count - len(df)
    print(f"   ✓ Valores nulos eliminados: {removed_count}")
    print(f"   ✓ Muestras restantes: {len(df)}")
    
    # 3. Crear mapeo de especialidades (Top N + Others)
    print(f"\n[3/6] Creando categorías de especialidades...")
    top_specialties = create_specialty_mapping(df, top_n=config.TOP_N_SPECIALTIES)
    print(f"   ✓ Top {config.TOP_N_SPECIALTIES} especialidades identificadas")
    print(f"   ✓ Total de categorías: {config.TOP_N_SPECIALTIES + 1} (incluyendo 'Others')")
    
    # 4. Aplicar chunking
    print(f"\n[4/6] Aplicando chunking (tamaño={config.CHUNK_SIZE}, overlap={config.OVERLAP})...")
    df_processed = process_dataset_with_chunks(
        df, 
        top_specialties,
        chunk_size=config.CHUNK_SIZE,
        overlap=config.OVERLAP
    )
    print(f"   ✓ Dataset original: {len(df)} muestras")
    print(f"   ✓ Dataset con chunks: {len(df_processed)} muestras")
    print(f"   ✓ Incremento: {len(df_processed) - len(df)} muestras adicionales")
    
    # 5. Train/test split
    print(f"\n[5/6] Realizando train/test split ({int((1-config.TEST_SIZE)*100)}/{int(config.TEST_SIZE*100)})...")
    if config.STRATIFY:
        stratify_column = df_processed[config.CLEAN_LABEL_COLUMN]
        print(f"   ✓ Usando split estratificado")
    else:
        stratify_column = None
    
    train_df, test_df = train_test_split(
        df_processed,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=stratify_column
    )
    print(f"   ✓ Train set: {len(train_df)} muestras")
    print(f"   ✓ Test set: {len(test_df)} muestras")

    tr_df, val_df = train_test_split(
        df_processed,
        test_size=config.VAL_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=stratify_column
    )
    print(f"   ✓ Train set final: {len(tr_df)} muestras")
    print(f"   ✓ Val set: {len(val_df)} muestras")
    
    # 6. Guardar datasets procesados
    print(f"\n[6/6] Guardando datasets procesados...")
    output_dir = Path(config.PROCESSED_DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    
    tr_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"   ✓ Train guardado en: {train_path}")
    print(f"   ✓ Val guardado en: {val_path}")
    print(f"   ✓ Test guardado en: {test_path}")
    
    # Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    print(f"Categorías: {df_processed[config.CLEAN_LABEL_COLUMN].nunique()}")
    print(f"\nDistribución en Train:")
    print(train_df[config.CLEAN_LABEL_COLUMN].value_counts().head(10))
    print(f"\n✅ Preparación de datos completada exitosamente!")
    print("=" * 60)


if __name__ == "__main__":
    main()
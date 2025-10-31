"""
Configuración centralizada para el proyecto Medical NLP MLOps
"""

# Rutas de datos
RAW_DATA_PATH = "data/raw/mtsamples.csv"
PROCESSED_DATA_DIR = "data/processed"

# Parámetros de preprocessing
TOP_N_SPECIALTIES = 19  # Top N especialidades + Others
CHUNK_SIZE = 2000  # Caracteres por chunk
OVERLAP = 200  # Solapamiento entre chunks

# Train/Test split
TEST_SIZE = 0.2  # 80/20 split
VAL_SIZE = 0.1  # 10% del train para validación
RANDOM_STATE = 786  # Semilla para reproducibilidad
STRATIFY = True  # Mantener proporción de clases

# Nombres de columnas
TEXT_COLUMN = "transcription"
LABEL_COLUMN = "medical_specialty"
CLEAN_LABEL_COLUMN = "specialty_clean"

# Models:
MODEL_NAME = "distilbert-base-uncased"
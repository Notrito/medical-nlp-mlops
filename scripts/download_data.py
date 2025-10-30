import kagglehub
import shutil
from pathlib import Path

# Download dataset
path = kagglehub.dataset_download("tboyle10/medicaltranscriptions")
print(f"Dataset descargado en: {path}")

# Mover a nuestra estructura
data_raw = Path("data/raw")
data_raw.mkdir(parents=True, exist_ok=True)

# Copiar archivos
shutil.copy(Path(path) / "mtsamples.csv", data_raw / "mtsamples.csv")
print(f"✅ Dataset copiado a {data_raw}")
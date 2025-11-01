#!/usr/bin/env python3
"""
tokenization.py

Script para tokenizar textos médicos y generar datasets listos
para entrenamiento con modelos tipo BERT (DistilBERT, BioBERT, etc.).

Uso:
    python src/tokenization.py \
        --model distilbert-base-uncased \
        --input data/processed \
        --output data/tokenized \
        --max_length 128
"""

import os
import argparse
import logging
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import pandas as pd


# ---------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Funciones principales
# ---------------------------------------------------------------------
def load_data(input_dir: str):
    """Carga train.csv y test.csv desde el directorio dado."""
    train_path = os.path.join(input_dir, "train.csv")
    val_path = os.path.join(input_dir, "val.csv")
    test_path = os.path.join(input_dir, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"No se encontraron train.csv o test.csv en {input_dir}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    logger.info(f"Cargados: {len(train_df)} train y {len(test_df)} test muestras.")
    return train_df, val_df, test_df


def tokenize_datasets(train_df, val_df, test_df, tokenizer, max_length):
    """Tokeniza los datasets y los devuelve como objetos de HuggingFace."""
    def tokenize_fn(batch):
        return tokenizer(
            batch["transcription"],
            truncation=True,
            padding=False,
            max_length=max_length
        )

    # Convertir a datasets HuggingFace
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    logger.info("Tokenizando datasets...")
    train_tok = train_dataset.map(tokenize_fn, batched=True)
    val_tok = val_dataset.map(tokenize_fn, batched=True)
    test_tok = test_dataset.map(tokenize_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Definir columnas a mantener
    keep_cols = ["input_ids", "attention_mask", "label"]
    train_tok.set_format(type="torch", columns=keep_cols)
    val_tok.set_format(type="torch", columns=keep_cols)
    test_tok.set_format(type="torch", columns=keep_cols)

    logger.info("Tokenización completada.")
    return train_tok, val_tok, test_tok


def save_datasets(train_tok, val_tok, test_tok, output_dir):
    """Guarda los datasets tokenizados en el directorio de salida."""
    os.makedirs(output_dir, exist_ok=True)

    train_tok.save_to_disk(os.path.join(output_dir, "train"))
    val_tok.save_to_disk(os.path.join(output_dir, "val"))
    test_tok.save_to_disk(os.path.join(output_dir, "test"))

    logger.info(f"Datasets guardados en: {output_dir}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Tokenización de textos médicos")
    parser.add_argument("--model", type=str, required=True, help="Nombre del modelo base (p. ej. distilbert-base-uncased o dmis-lab/biobert-base-cased-v1.2)")
    parser.add_argument("--input", type=str, required=True, help="Directorio con train.csv y test.csv")
    parser.add_argument("--output", type=str, required=True, help="Directorio donde guardar los datasets tokenizados")
    parser.add_argument("--max_length", type=int, default=128, help="Longitud máxima de secuencia")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Proporción del conjunto de validación")

    args = parser.parse_args()

    # Cargar tokenizer
    logger.info(f"Cargando tokenizer de {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Cargar datos
    train_df, val_df, test_df = load_data(args.input)

    # Tokenizar
    train_tok, val_tok, test_tok = tokenize_datasets(
        train_df, val_df, test_df, tokenizer, args.max_length
    )

    # Guardar
    save_datasets(train_tok, val_tok, test_tok, args.output)

    logger.info("✅ Proceso de tokenización finalizado correctamente.")


if __name__ == "__main__":
    main()

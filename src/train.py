#!/usr/bin/env python3
"""
train.py

Script para entrenar modelos de clasificación de especialidades médicas
con integración de MLflow para tracking de experimentos.

Uso:
    python src/train.py \
        --model distilbert-base-uncased \
        --data_dir data/tokenized \
        --output_dir models/distilbert \
        --epochs 3 \
        --batch_size 8
"""

import os
import argparse
import logging
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_from_disk
import mlflow
import mlflow.pytorch
import mlflow.transformers
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import numpy as np
from accelerate import Accelerator


# ---------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def load_tokenized_data(data_dir):
    """
    Carga los datasets tokenizados desde disco.
    
    Args:
        data_dir: Directorio donde están train/, val/, test/
        
    Returns:
        Tupla (train_dataset, val_dataset, test_dataset)
    """
    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")
    test_path = os.path.join(data_dir, "test")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"No se encontró {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"No se encontró {val_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"No se encontró {test_path}")
    
    logger.info(f"Cargando datasets desde {data_dir}...")
    train_dataset = load_from_disk(train_path)
    val_dataset = load_from_disk(val_path)
    test_dataset = load_from_disk(test_path)
    
    logger.info(f"✓ Train: {len(train_dataset)} muestras")
    logger.info(f"✓ Val: {len(val_dataset)} muestras")
    logger.info(f"✓ Test: {len(test_dataset)} muestras")

    return train_dataset, val_dataset, test_dataset


def compute_metrics(eval_pred):
    """Calcula métricas de evaluación."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calcular métricas
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    precision, recall, _, _ = precision_recall_fscore_support(
        labels, predictions, average='macro'
    )
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall
    }

def train_model(args):
    """Función principal para entrenar el modelo."""
    
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("medical-nlp-classification")
    
    with mlflow.start_run(run_name=f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Logear hiperparámetros
        mlflow.log_params({
            "model_name": args.model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "fp16": args.fp16,
            "gradient_accumulation_steps": args.gradient_accumulation_steps
        })
     
        # Cargar datasets tokenizados
        train_dataset, val_dataset, test_dataset = load_tokenized_data(args.data_dir)
        num_labels = len(set(train_dataset['label']))
        logger.info(f"Número de etiquetas: {num_labels}")
        mlflow.log_param("num_labels", num_labels)

        # Cargar modelo y tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=num_labels
        )
        logger.info(f"Modelo {args.model} cargado con éxito.")

        # data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Configurar argumentos de entrenamiento
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            fp16=args.fp16,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            logging_dir=os.path.join(args.output_dir, "logs"),
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            report_to="mlflow"
        )
        last_checkpoint = get_last_checkpoint(args.output_dir)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            data_collator=data_collator
        )

        # entrenar
        logger.info("Iniciando entrenamiento...")
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        logger.info("Entrenamiento completado.")

        # log metrics
        mlflow.log_metrics({
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics['train_runtime'],
            "train_samples_per_second": train_result.metrics['train_samples_per_second']
        })

        # Evaluar en conjunto de validación
        logger.info("📊 Evaluando en conjunto de validación...")
        val_metrics = trainer.evaluate()
        mlflow.log_metrics({
            f"val_{k}": v for k, v in val_metrics.items()
        })

        # Evaluar en conjunto de test
        logger.info("📊 Evaluando en conjunto de test...")
        test_metrics = trainer.evaluate(test_dataset)
        mlflow.log_metrics({
            f"test_{k}": v for k, v in test_metrics.items()
        })

        # Guardar modelo
        logger.info(f"💾 Guardando modelo en {args.output_dir}")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        # Unwrap model antes de guardar en MLflow (necesario con fp16)
        mlflow.transformers.log_model(
            transformers_model={"model": trainer.model, "tokenizer": tokenizer},
            artifact_path="model",
            task="text-classification"
        )
        logger.info("✅ Modelo guardado en MLflow correctamente")
        
        logger.info("✅ Entrenamiento completado exitosamente!")
        logger.info(f"   - Val F1 (macro): {val_metrics['eval_f1_macro']:.4f}")
        logger.info(f"   - Test F1 (macro): {test_metrics['eval_f1_macro']:.4f}")


# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de modelo de clasificación médica")
    
    # Argumentos del modelo
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Nombre del modelo base (ej: distilbert-base-uncased, dmis-lab/biobert-base-cased-v1.2)"
    )
    
    # Argumentos de datos
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directorio con datasets tokenizados (train/, val/, test/)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directorio donde guardar el modelo entrenado"
    )
    
    # Hiperparámetros de entrenamiento
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Número de épocas de entrenamiento"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Tamaño del batch (ajustar según GPU disponible)"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate para el optimizador"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Longitud máxima de secuencia (debe coincidir con tokenización)"
    )
    
    # Optimizaciones para GPU
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Usar mixed precision training (fp16) para ahorrar memoria"
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Pasos de acumulación de gradiente (simula batch más grande)"
    )
    
    args = parser.parse_args()
    
    # Crear directorio de salida si no existe
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Ejecutar entrenamiento
    train_model(args)


if __name__ == "__main__":
    main()
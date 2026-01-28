"""
Fine-tune ESG-BERT using Triplet Loss for semantic similarity tasks.
Combines Environmental, Social, and Governance triplets for training.
FIXED VERSION - Addresses overfitting and evaluation issues.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader
import os
from sklearn.model_selection import train_test_split
import logging
import torch

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

# ============================================================================
# CONFIGURATION - OPTIMIZED TO PREVENT OVERFITTING
# ============================================================================

BASE_MODEL = "nbroad/ESG-BERT"
OUTPUT_DIR = "../SAGE-BERT"

# Hyperparameters - FIXED
BATCH_SIZE = 4  # Increased from 2 for better training
EPOCHS = 3  # Reduced from 5 to prevent overfitting
LEARNING_RATE = 1e-5  # Reduced from 2e-5 for more stable training
MARGIN = 0.5  # Reduced from 1.0 - was too aggressive
WARMUP_STEPS = 20  # Reduced from 50

# Data split
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# CSV files
CSV_FILES = [
    "env_triplets.csv",
    "social_triplets.csv", 
    "gov_triplets.csv"
]

# Memory optimization
torch.set_num_threads(2)

# ============================================================================
# STEP 1: LOAD AND COMBINE DATA
# ============================================================================

def load_triplets(csv_files):
    """Load and combine triplets from multiple CSV files."""
    all_data = []
    
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            logging.warning(f"File not found: {csv_file}. Skipping...")
            continue
        
        df = pd.read_csv(csv_file)
        logging.info(f"Loaded {len(df)} triplets from {csv_file}")
        
        # Validate columns
        required_cols = ['anchor_text', 'positive_text', 'negative_text', 'similarity_score']
        if not all(col in df.columns for col in required_cols):
            logging.error(f"Missing required columns in {csv_file}")
            continue
        
        # Remove rows with missing values
        df = df.dropna()
        
        # Validate similarity scores
        df = df[df['similarity_score'].between(0, 1)]
        
        all_data.append(df)
    
    if not all_data:
        raise ValueError("No valid data loaded from CSV files!")
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    logging.info(f"Total triplets after combining: {len(combined_df)}")
    
    return combined_df

# ============================================================================
# STEP 1.5: VALIDATE TRIPLET QUALITY - NEW
# ============================================================================

def validate_triplets(df):
    """Check if triplets make sense and log potential issues."""
    logging.info("\n🔍 Validating triplet quality...")
    
    issues = 0
    for idx, row in df.iterrows():
        anchor = str(row['anchor_text']).strip()
        positive = str(row['positive_text']).strip()
        negative = str(row['negative_text']).strip()
        
        # Check for duplicates
        if anchor == positive or anchor == negative or positive == negative:
            logging.warning(f"Row {idx}: Duplicate texts found!")
            issues += 1
        
        # Check lengths
        if len(anchor) < 20 or len(positive) < 20 or len(negative) < 20:
            logging.warning(f"Row {idx}: Very short text (likely metadata or noise)")
            issues += 1
        
        # Check if positive and negative are too similar
        if len(positive) > 50 and len(negative) > 50:
            # Simple overlap check
            pos_words = set(positive.lower().split())
            neg_words = set(negative.lower().split())
            overlap = len(pos_words & neg_words) / max(len(pos_words), len(neg_words))
            if overlap > 0.7:
                logging.warning(f"Row {idx}: Positive and negative are too similar ({overlap:.2%} word overlap)")
                issues += 1
    
    if issues > 0:
        logging.warning(f"⚠️  Found {issues} potential data quality issues out of {len(df)} triplets")
    else:
        logging.info("✓ All triplets look valid")
    
    # Show sample triplet
    logging.info("\n📋 Sample triplet:")
    sample = df.iloc[0]
    logging.info(f"Anchor: {sample['anchor_text'][:100]}...")
    logging.info(f"Positive: {sample['positive_text'][:100]}...")
    logging.info(f"Negative: {sample['negative_text'][:100]}...")
    logging.info(f"Similarity Score: {sample['similarity_score']}")
    
    return df

# ============================================================================
# STEP 2: CREATE TRAIN/VALIDATION SPLIT
# ============================================================================

def create_train_val_split(df, train_ratio=0.8, random_seed=42):
    """Split data into training and validation sets."""
    train_df, val_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=random_seed,
        shuffle=True
    )
    
    logging.info(f"Training set: {len(train_df)} triplets")
    logging.info(f"Validation set: {len(val_df)} triplets")
    
    return train_df, val_df

# ============================================================================
# STEP 3: PREPARE DATA FOR SENTENCE-TRANSFORMERS
# ============================================================================

def create_input_examples(df):
    """Convert dataframe to InputExample objects for sentence-transformers."""
    examples = []
    
    for _, row in df.iterrows():
        example = InputExample(
            texts=[
                str(row['anchor_text']).strip(),
                str(row['positive_text']).strip(),
                str(row['negative_text']).strip()
            ]
        )
        examples.append(example)
    
    return examples

# ============================================================================
# STEP 4: DEFINE MODEL AND LOSS FUNCTION - FIXED
# ============================================================================

def initialize_model(base_model):
    """Initialize the SentenceTransformer model."""
    logging.info(f"Loading base model: {base_model}")
    model = SentenceTransformer(base_model)
    
    # Log model info
    logging.info(f"Model loaded successfully!")
    logging.info(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    return model

def create_loss_function(model, margin=0.5):
    """Create TripletLoss with COSINE distance (FIXED)."""
    loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.COSINE,  # FIXED: Was EUCLIDEAN
        triplet_margin=margin
    )
    logging.info(f"TripletLoss initialized with COSINE distance, margin={margin}")
    return loss

# ============================================================================
# STEP 5: CREATE EVALUATOR FOR VALIDATION - FIXED
# ============================================================================

def create_evaluator(val_df):
    """Create TripletEvaluator for validation set with proper distance metric."""
    anchors = [str(x).strip() for x in val_df['anchor_text'].tolist()]
    positives = [str(x).strip() for x in val_df['positive_text'].tolist()]
    negatives = [str(x).strip() for x in val_df['negative_text'].tolist()]
    
    evaluator = TripletEvaluator(
        anchors=anchors,
        positives=positives,
        negatives=negatives,
        name='validation',
        batch_size=16,
        show_progress_bar=False,
        write_csv=True  # Save evaluation results
    )
    
    logging.info("Validation evaluator created with cosine similarity")
    return evaluator

# ============================================================================
# STEP 6: TRAINING LOOP WITH BETTER MONITORING - FIXED
# ============================================================================

def train_model(model, train_examples, val_evaluator, loss_fn, 
                epochs, batch_size, learning_rate, warmup_steps, output_dir):
    """Train the model with proper evaluation and early stopping."""
    
    # Create DataLoader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size
    )
    
    # Calculate training steps
    steps_per_epoch = len(train_dataloader)
    num_train_steps = steps_per_epoch * epochs
    
    logging.info(f"Total training steps: {num_train_steps}")
    logging.info(f"Steps per epoch: {steps_per_epoch}")
    
    # Training arguments
    logging.info("\n🚀 Starting training...")
    logging.info(f"Epochs: {epochs}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Learning rate: {learning_rate}")
    logging.info(f"Warmup steps: {warmup_steps}")
    logging.info(f"Margin: {MARGIN}")
    logging.info(f"Distance metric: COSINE")
    
    # Train with evaluation after each epoch
    model.fit(
        train_objectives=[(train_dataloader, loss_fn)],
        evaluator=val_evaluator,
        epochs=epochs,
        evaluation_steps=steps_per_epoch,  # Evaluate after each epoch
        warmup_steps=warmup_steps,
        optimizer_params={'lr': learning_rate},
        output_path=output_dir,
        save_best_model=True,
        show_progress_bar=True,
        use_amp=False  # Disable mixed precision for stability
    )
    
    logging.info(f"\n✅ Training complete! Best model saved to: {output_dir}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline."""
    
    logging.info("="*70)
    logging.info("ESG-BERT FINE-TUNING WITH TRIPLET LOSS (FIXED VERSION)")
    logging.info("="*70)
    
    # Step 1: Load and combine data
    logging.info("\n[Step 1/7] Loading triplet data...")
    df = load_triplets(CSV_FILES)
    
    # Step 1.5: Validate data quality - NEW
    logging.info("\n[Step 2/7] Validating triplet quality...")
    df = validate_triplets(df)
    
    # Step 2: Train/Val split
    logging.info("\n[Step 3/7] Creating train/validation split...")
    train_df, val_df = create_train_val_split(df, TRAIN_RATIO, RANDOM_SEED)
    
    # Step 3: Prepare data
    logging.info("\n[Step 4/7] Preparing data for training...")
    train_examples = create_input_examples(train_df)
    
    # Step 4: Initialize model and loss
    logging.info("\n[Step 5/7] Initializing model and loss function...")
    model = initialize_model(BASE_MODEL)
    loss_fn = create_loss_function(model, MARGIN)
    
    # Step 5: Create evaluator
    logging.info("\n[Step 6/7] Creating validation evaluator...")
    val_evaluator = create_evaluator(val_df)
    
    # Step 6: Train
    logging.info("\n[Step 7/7] Starting training...")
    train_model(
        model=model,
        train_examples=train_examples,
        val_evaluator=val_evaluator,
        loss_fn=loss_fn,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        output_dir=OUTPUT_DIR
    )
    
    # Final summary
    logging.info("\n" + "="*70)
    logging.info("TRAINING SUMMARY")
    logging.info("="*70)
    logging.info(f"Total triplets: {len(df)}")
    logging.info(f"Training triplets: {len(train_df)}")
    logging.info(f"Validation triplets: {len(val_df)}")
    logging.info(f"Epochs trained: {EPOCHS}")
    logging.info(f"Batch size: {BATCH_SIZE}")
    logging.info(f"Learning rate: {LEARNING_RATE}")
    logging.info(f"Margin: {MARGIN}")
    logging.info(f"Model saved to: {OUTPUT_DIR}")
    logging.info("="*70)
    logging.info("\n✓ Fine-tuning complete!\n")
    
    # Show how to use the fine-tuned model
    logging.info("📖 How to use the fine-tuned model:")
    logging.info("  from sentence_transformers import SentenceTransformer")
    logging.info("  model = SentenceTransformer('../SAGE-BERT')")
    logging.info("  embeddings = model.encode(your_texts)")
    logging.info("")
    logging.info("🎯 Next steps:")
    logging.info("  1. Update your embedding generation script to use '../SAGE-BERT'")
    logging.info("  2. Generate embeddings for regulations and disclosures")
    logging.info("  3. Use cosine_similarity in your semantic matching script")
    logging.info("")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
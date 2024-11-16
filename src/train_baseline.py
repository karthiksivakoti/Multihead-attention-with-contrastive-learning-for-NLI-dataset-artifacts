import torch
from pathlib import Path
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from model import NLIModelWithDebias
from trainer import Trainer
from model_utils import prepare_batch_for_gpu
from data_handler import get_dataloaders, SNLIProcessor
import json
import logging
import argparse
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='google/electra-small-discriminator')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--output_dir', type=str, default='baseline_model')
    return parser.parse_args()
def setup_logging(output_dir: Path):
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('training')
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    logger.info(f"Using device: {device}")
    processor = SNLIProcessor()
    if not (Path('data/processed') / 'snli_train.json').exists():
        logger.info("Processing SNLI dataset...")
        processor.process_dataset()
    with open('src/config.json') as f:
        config = json.load(f)
    config['training']['num_epochs'] = args.num_epochs
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader, val_loader = get_dataloaders(
        tokenizer=tokenizer,
        batch_size=args.batch_size
    )
    model = NLIModelWithDebias(args.model_name)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=config['training']['weight_decay']
    )
    num_training_steps = len(train_loader) * 3
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['training']['warmup_steps'],
        num_training_steps=len(train_loader) * args.num_epochs
    )
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=args.output_dir,
        config=config,
        is_baseline=True,
        save_artifacts=True,
        keep_best_checkpoint_only=True
    )
    logger.info("Starting baseline training...")
    trainer.train()
    logger.info("Baseline training completed!")

if __name__ == "__main__":
    main()

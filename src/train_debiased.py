import torch
from pathlib import Path
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from model import NLIModelWithDebias
from trainer import Trainer
from data_handler import get_dataloaders
import json
import logging
import argparse
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='google/electra-small-discriminator')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--output_dir', type=str, default='debiased_model')
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
def find_latest_checkpoint(output_dir: Path):
    checkpoints = [
        cp for cp in output_dir.iterdir()
        if cp.is_dir() and cp.name.startswith('checkpoint-') and cp.name[10:].isdigit()
    ]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split('-')[1]))
        checkpoint_path = latest_checkpoint / 'model.pt'
        if checkpoint_path.exists():
            step_num = int(latest_checkpoint.name.split('-')[1])
            epoch = step_num // 500
            return epoch, checkpoint_path
    best_model_path = output_dir / 'best_model' / 'model.pt'
    if best_model_path.exists():
        return 0, best_model_path
    return 0, None

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    logger.info(f"Using device: {device}")
    with open('src/config.json') as f:
        config = json.load(f)
    config['training']['num_epochs'] = args.num_epochs
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader, val_loader = get_dataloaders(tokenizer=tokenizer, batch_size=args.batch_size)
    model = NLIModelWithDebias(args.model_name)
    checkpoint_dir = output_dir / 'checkpoints'
    start_epoch, latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        logger.info(f"Loading checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if start_epoch == -1:
            start_epoch = 0
        logger.info(f"Continuing training from epoch {start_epoch}")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=config['training'].get('weight_decay', 0.01))
    num_training_steps = len(train_loader) * args.num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, scheduler=scheduler, device=device, output_dir=args.output_dir, config=config, is_baseline=False, save_artifacts=True, keep_best_checkpoint_only=True)
    logger.info("Starting/Resuming debiased training...")
    trainer.train()
    logger.info("Training completed!")

if __name__ == "__main__":
    main()

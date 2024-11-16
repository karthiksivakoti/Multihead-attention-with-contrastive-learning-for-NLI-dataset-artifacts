import torch
from transformers import AutoTokenizer
from model import NLIModelWithDebias
from data_handler import get_dataloaders
from trainer import Trainer
from helpers import set_seed, prepare_output_directories, setup_logger, save_experiment_config
from pathlib import Path
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Train NLI model with debiasing')
    parser.add_argument('--model_name', type=str, default='google/electra-small-discriminator', help='Base model to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='results/debiased', help='Output directory')
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    directories = prepare_output_directories(args.output_dir)
    logger = setup_logger('debiasing_training', log_file=directories['base'] / 'training.log')
    save_experiment_config(vars(args), directories['base'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader, val_loader = get_dataloaders(tokenizer=tokenizer, batch_size=args.batch_size)
    model = NLIModelWithDebias(args.model_name)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs
    )
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed!")

if __name__ == "__main__":
    main()

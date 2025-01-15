# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import pandas as pd
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
Model training main program
"""

# Set seeds for reproducibility
seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main(config):
    # Create a directory to save the model
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # Loading training data
    train_data = load_data(config["train_data_path"], config)
    # Loading the model
    model = TorchModel(config)
    # Indicates whether to use GPU
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("GPU is available, migrate the model to GPU")
        model = model.cuda()
    # Loading Optimizer
    optimizer = choose_optimizer(config, model)
    # Loading effect test class
    evaluator = Evaluator(config, model, logger)
    # 训练
    for epoch in range(config["epoch"]):
        model.train()
        logger.info("Epoch %d begin" % (epoch + 1))
        train_loss = []

        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() if isinstance(d, torch.Tensor) else d for d in batch_data]

            optimizer.zero_grad()
            # Unpack batch data
            try:
                input_ids, attention_mask, labels = batch_data
            except ValueError as e:
                logger.error(f"Unexpected batch_data format at index {index}: {batch_data}. Error: {e}")
                continue

            # Forward pass
            loss = model(input_ids, attention_mask, labels)
            if not isinstance(loss, torch.Tensor):
                logger.error(f"Loss is not a tensor: {loss}")
                continue

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % max(1, int(len(train_data) / 2)) == 0:
                logger.info("Batch loss %f" % loss.item())

        logger.info("Epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)

        # Save model after each epoch
        model_path = os.path.join(config["model_path"], "epoch_%d.pth" % (epoch + 1))
        torch.save(model.state_dict(), model_path)

    return acc

if __name__ == "__main__":
    # DataFrame to store results
    results_df = pd.DataFrame(
        columns=["model_type", "learning_rate", "hidden_size", "batch_size", "pooling_style", "acc"])

    # Hyperparameter grid search
    for model_type in ["gated_cnn", "fast_text", "lstm"]:
        Config["model_type"] = model_type
        for lr in [1e-3, 1e-4]:
            Config["learning_rate"] = lr
            for hidden_size in [128, 256]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 256]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg", "max"]:
                        Config["pooling_style"] = pooling_style

                        logger.info(f"Starting training with config: {Config}")
                        acc = main(Config)

                        # Append results to the DataFrame
                        results_df = results_df._append({
                            "model_type": model_type,
                            "learning_rate": lr,
                            "hidden_size": hidden_size,
                            "batch_size": batch_size,
                            "pooling_style": pooling_style,
                            "acc": acc
                        }, ignore_index=True)

    # Save results to Excel
    results_path = "results.xlsx"
    results_df.to_excel(results_path, index=False)
    logger.info(f"Results saved to {results_path}")

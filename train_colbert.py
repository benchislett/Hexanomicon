import json, jsonschema
import time
import tqdm
import numpy as np
import openai
import asyncio
from ragatouille import RAGTrainer

from cards import CardDataset

async def main():

    dataset = CardDataset.from_file("oracle-cards-20250405210637.json")

    trainer = RAGTrainer(model_name="MTGColBERTv_0_0", pretrained_model_name="colbert-ir/colbertv2.0", language_code="en")

    with open("train_triples_easy_med_10x.jsonl", "r") as f:
        train_triples = []
        for line in f:
            data = json.loads(line)
            train_triples.append((data["query"], data["positive"], data["negative"]))

    print(f"Loaded {len(train_triples)} training triples.")

    trainer.prepare_training_data(raw_data=train_triples, data_out_path="./train_data_v0_0/", all_documents=dataset.formatted_cards.values(), mine_hard_negatives=False)

    trainer.train(batch_size=32,
                nbits=4, # How many bits will the trained model use when compressing indexes
                maxsteps=100_000, # Maximum steps hard stop
                use_ib_negatives=True, # Use in-batch negative to calculate loss
                dim=128, # How many dimensions per embedding. 128 is the default and works well.
                learning_rate=5e-6, # Learning rate, small values ([3e-6,3e-5] work best if the base model is BERT-like, 5e-6 is often the sweet spot)
                doc_maxlen=256, # Maximum document length. Because of how ColBERT works, smaller chunks (128-256) work very well.
                use_relu=False, # Disable ReLU -- doesn't improve performance
                warmup_steps="auto", # Defaults to 10%
    )

if __name__ == '__main__':
    asyncio.run(main())
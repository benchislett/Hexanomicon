import json, jsonschema
import time
import tqdm
import numpy as np
import openai
import asyncio
from ragatouille import RAGTrainer

with open("oracle-cards-20250405210637.json", "r") as f:
    data = json.load(f)
print(f"Loaded {len(data)} cards")

def get_card_data(oracle_data):
    if type(oracle_data) != dict:
        print(f"Invalid oracle data: {oracle_data}")
        return None
    if "memorabilia" in oracle_data.get("set_type", ""):
        # skip memorabilia cards
        return None
    if "//" in oracle_data.get("name", ""):
        # todo: handle double-faced cards separately
        return None
    if "mana_cost" not in oracle_data or oracle_data["mana_cost"] is None:
        print(f"Missing mana_cost for {oracle_data['name']}")
        return None
    if "oracle_text" not in oracle_data or oracle_data["oracle_text"] is None:
        print(f"Missing oracle_text for {oracle_data['name']}")
        return None
    
    # check if legal in some format
    # legal_formats = oracle_data.get("legalities", []).items()
    # legal_formats = [f for f, l in legal_formats if l == "legal"]
    # if not legal_formats:
    #     print(f"Not legal in any format for {oracle_data['name']}")
    #     return None
    return {
        "name": oracle_data["name"],
        "mana_cost": oracle_data["mana_cost"],
        "cmc": oracle_data["cmc"],
        "type": oracle_data["type_line"],
        "text": oracle_data["oracle_text"],
        "power": oracle_data.get("power"),
        "toughness": oracle_data.get("toughness"),
        "loyalty": oracle_data.get("loyalty"),
        "colors": oracle_data.get("colors"),
        "keywords": oracle_data.get("keywords"),
    }

all_cards_by_name = {card["name"]: get_card_data(card) for card in data}
all_cards_by_name = {k: v for k, v in all_cards_by_name.items() if v is not None}

sample_cards = list(all_cards_by_name.keys())
print(len(sample_cards), "sample cards")

def format_card(card_data):
    """Format a card for input into a vector-embedding model."""
    colors = list(map(lambda ch: {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green"}.get(ch, ch), card_data.get("colors", [])))

    # header = "The following is a card from the game Magic: The Gathering.\n"
    header = ""
    prompt = f"""{header}
{card_data["name"]}.
{card_data["type"]}.
Mana cost: {card_data["mana_cost"]}
Converted mana cost: {card_data["cmc"]}
Colors: {", ".join(colors) or "Colorless"}.
Oracle text: {card_data["text"] or "None"}
"""
    if card_data["loyalty"] is not None:
        prompt += f"""Loyalty: {card_data["loyalty"]}
"""

    if card_data["power"] is not None and card_data["toughness"] is not None:
        prompt += f"""Power: {card_data["power"]}
Toughness: {card_data["toughness"]}
"""
    if card_data["keywords"] is not None and len(card_data["keywords"]) > 0:
        prompt += f"""Keywords: {", ".join(card_data["keywords"])}
"""
    return prompt.strip()

# sample_cards = list(all_cards_by_name.keys())
formatted_cards = [format_card(all_cards_by_name[card]) for card in sample_cards]
postprocessed_formatted_cards = formatted_cards

trainer = RAGTrainer(model_name="MTGColBERTv_0_0", pretrained_model_name="colbert-ir/colbertv2.0", language_code="en")

import json

with open("train_triples_easy_med.jsonl", "r") as f:
    train_triples = []
    for line in f:
        data = json.loads(line)
        train_triples.append((data["query"], data["positive"], data["negative"]))

print(f"Loaded {len(train_triples)} training triples.")

trainer.prepare_training_data(raw_data=train_triples, data_out_path="./train_data_v0_0/", all_documents=formatted_cards, mine_hard_negatives=False)

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


import json, jsonschema
import time
import tqdm
import numpy as np
import openai
import asyncio
from ragatouille import RAGPretrainedModel


async def main():

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

    # RAG = RAGPretrainedModel.from_pretrained(".ragatouille/colbert/none/2025-04/30/18.51.27/checkpoints/colbert-100000")

    # documents = formatted_cards
    # document_ids = sample_cards
    # index_path = RAG.index(
    #     index_name="mtg_cards_ft",
    #     collection=documents,
    #     document_ids=document_ids,
    #     split_documents=False
    # )

    RAG = RAGPretrainedModel.from_index(".ragatouille/colbert/indexes/mtg_cards_ft")

    query = "cheap instant-speed cantrips"
    res = RAG.search(query, k=100)

    print(f"Found {len(res)} results")
    for doc in res[:100]:
        print(f"Document: {doc['document_id']}")
        print(f"Score: {doc['score']}")
        print(f"Document: {doc}")
        print("-" * 80)

if __name__ == '__main__':
    asyncio.run(main())
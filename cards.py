import json
from functools import lru_cache
from dataclasses import dataclass
from typing import Optional
from fire import Fire
import numpy as np
import textwrap
import logging, sys

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s [%(name)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr)],
)

logger = logging.getLogger(__name__)

@lru_cache(maxsize=None)
def log_once(message: str):
    """Log a message only once."""
    logger.info(message)


@dataclass
class CardData:
    name: str

    mana_cost: str
    """Formatted mana cost: "{U}{W}{R}" for example"""

    converted_mana_cost: float
    type_line: str
    base_type: str
    oracle_text: str
    power: Optional[int]
    toughness: Optional[int]
    loyalty: Optional[int]
    colors: list[str]
    keywords: list[str]

    @staticmethod
    def from_oracle_dict(oracle_data: dict) -> Optional["CardData"]:
        """Create a CardData object from the oracle data object."""
        return create_card_data_from_oracle_dict(oracle_data)

def get_base_type(type_line: str) -> str:
    """Get the base type of a card from its type line."""
    type_line = type_line.lower()
    if "-" in type_line:
        type_line = type_line.split("-")[0]
    
    if "creature" in type_line:
        return "Creature"
    elif "enchantment" in type_line:
        return "Enchantment"
    elif "artifact" in type_line:
        return "Artifact"
    elif "planeswalker" in type_line:
        return "Planeswalker"
    elif "land" in type_line:
        return "Land"
    elif "sorcery" in type_line:
        return "Sorcery"
    elif "instant" in type_line:
        return "Instant"
    elif "battle" in type_line:
        return "Battle"
    else:
        return "Other"


def create_card_data_from_oracle_dict(oracle_data: dict) -> Optional[CardData]:
    """Create a CardData object from the oracle data object."""
    if type(oracle_data) != dict:
        print(f"Invalid oracle data: {oracle_data}")
        return None
    if "memorabilia" in oracle_data.get("set_type", ""):
        # skip memorabilia cards
        return None
    if "//" in oracle_data.get("name", ""):
        log_once("Double-faced cards not yet supported, omitting.")
        # todo: handle double-faced cards separately
        return None
    if "mana_cost" not in oracle_data or oracle_data["mana_cost"] is None:
        print(f"Missing mana_cost for {oracle_data['name']}")
        return None
    if "oracle_text" not in oracle_data or oracle_data["oracle_text"] is None:
        print(f"Missing oracle_text for {oracle_data['name']}")
        return None
    
    color_mapping = {
        "W": "White",
        "U": "Blue",
        "B": "Black",
        "R": "Red",
        "G": "Green",
    }
    if "colors" in oracle_data:
        oracle_data["colors"] = [color_mapping.get(color, color) for color in oracle_data["colors"]]

    if get_base_type(oracle_data["type_line"]) == "Other":
        # skip cards with unknown/disallowed base type
        return None
    
    return CardData(
        name=oracle_data["name"],
        mana_cost=oracle_data["mana_cost"],
        converted_mana_cost=oracle_data["cmc"],
        type_line=oracle_data["type_line"],
        base_type=get_base_type(oracle_data["type_line"]),
        oracle_text=oracle_data["oracle_text"],
        power=oracle_data.get("power"),
        toughness=oracle_data.get("toughness"),
        loyalty=oracle_data.get("loyalty"),
        colors=oracle_data.get("colors", []),
        keywords=oracle_data.get("keywords", []),
    )

def load_cards_from_oracle_dataset(file: str) -> dict[str, CardData]:
    """Load cards from the oracle dataset."""

    with open(file, "r") as f:
        data = json.load(f)
    
    cards = {}
    for oracle_card_data in data:
        card_data = CardData.from_oracle_dict(oracle_card_data)
        if card_data is not None:
            cards[card_data.name] = card_data

    logger.info(f"Imported {len(cards)} of {len(data)} cards from {file}")

    return cards


def format_card(card: CardData, include_context_header = False) -> str:
    """Format a card for text output."""
    header = ""
    if include_context_header:
        header = "The following is a card from the game Magic: The Gathering.\n"

    prompt = f"""{header}
{card.name}
{card.type_line}
Colors: {", ".join(card.colors) or "Colorless"}
""".strip()
    if card.mana_cost != "":
        prompt += f"""
Mana cost: {card.mana_cost}"""
    prompt += f"""
Converted mana cost: {card.converted_mana_cost}"""
    if card.loyalty is not None:
        prompt += f"""
Loyalty: {card.loyalty}"""

    if card.power is not None and card.toughness is not None:
        prompt += f"""
Power/Toughness: {card.power}/{card.toughness}"""
    if card.keywords:
        prompt += f"""
Keywords: {", ".join(card.keywords)}"""
    prompt += f"""
Oracle text: {card.oracle_text or "None"}"""
    return prompt.strip()


@dataclass
class CardDataset:
    """A dataset of cards, indexed by name."""
    card_data: dict[str, CardData]
    formatted_cards: dict[str, str]

    @staticmethod
    def from_file(file: str) -> "CardDataset":
        """Load a dataset from a file."""
        card_data = load_cards_from_oracle_dataset(file)
        formatted_cards = {name: format_card(card) for name, card in card_data.items()}
        return CardDataset(card_data, formatted_cards)

    def __len__(self):
        """Return the number of cards in the dataset."""
        return len(self.card_data)


def wrap_preserve(text: str, width: int = 80) -> str:
    return "\n".join(
        "\n".join(textwrap.wrap(line, width, replace_whitespace=False, drop_whitespace=False))
        if line else ""                       # keep blank lines
        for line in text.splitlines()
    )


def main(file: str = "oracle-cards-20250405210637.json", seed: int = 0):
    dataset = CardDataset.from_file(file)

    # sample a card of each base type
    types_seen = set()
    sampled_cards = []
    iter_idxs = np.arange(len(dataset))
    np.random.seed(seed)
    np.random.shuffle(iter_idxs)

    card_data_list = list(dataset.card_data.values())
    for i in iter_idxs:
        card = card_data_list[i]
        if card.base_type not in types_seen:
            types_seen.add(card.base_type)
            sampled_cards.append(card.name)
    
    print("\n" + "=" * 80 + "\n")
    for card_name in sampled_cards:
        print(wrap_preserve(dataset.formatted_cards[card_name]))
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    Fire(main)

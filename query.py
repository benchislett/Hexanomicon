import json, jsonschema
import time
import tqdm
import numpy as np
import openai
import asyncio
from ragatouille import RAGPretrainedModel

from cards import CardDataset

class ScopedTimer:
    def __init__(self, name: str):
        self.name = name
        self.start_time = time.time()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        print(f"{self.name}: {elapsed_time:.2f} seconds")

async def main():

    dataset = CardDataset.from_file("oracle-cards-20250405210637.json")

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

    while True:
        query = input("Enter a query: ")
        if query.lower() == "exit":
            break

        with ScopedTimer("Search"):
            res = RAG.search(query, k=10)

        print(f"Found {len(res)} results")
        for doc in res:
            print(f"Document: {doc['document_id']}")
            print(f"Score: {doc['score']}")
            print(f"Document: {doc}")
            print("-" * 80)

if __name__ == '__main__':
    asyncio.run(main())
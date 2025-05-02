import json, jsonschema
import time
import tqdm
import numpy as np
import openai
import asyncio
import os
import logging

from cards import CardDataset

logger = logging.getLogger(__name__)


def prepare_synthetic_querygen_prompt(formatted_card: str) -> str:
    prompt = """You are an expert AI assisting in creating a high-quality, diverse synthetic dataset to train Information Retrieval systems for Magic: The Gathering cards.
Your task is to analyse the card description below and generate a set of rich, high-quality potential queries for which the given card would rank very highly.

The output queries should be diverse, covering various aspects of the card, including its mechanics, roles, strategies, and synergies. The queries should be in natural language and should not be too similar to each other.
They should feature a variety of keywords and phrases that a user might use when searching for cards like this one, including specific jargons or slang used in the Magic: The Gathering community. Avoid naming the card directly or repeating its exact text.
You should submit about 5-10 brief queries. Include at least two queries that are short collections of keywords or phrases, and at least two queries that are full sentences.
Your output should be a JSON object with the same schema as the following example:
<schema>
{
    "hypothetical_queries": ["<query1>", "<query2>", "<query3>", "<query4>", "<query5>", "<query6>"]
}
</schema>""".strip() + f"""
<input>
{formatted_card}
</input>
"""
    return prompt.strip()
    
def postprocess_synthetic_querygen_response(model_output: str) -> list[str]:
    try:
        model_output = model_output.strip().strip("```").strip().strip("json").strip()
        data = json.loads(model_output)
        schema = { "type": "object", "properties": { "hypothetical_queries": { "type": "array", "items": { "type": "string" } } }, "required": ["hypothetical_queries"] }
        jsonschema.validate(data, schema)
        return data["hypothetical_queries"]
    except json.JSONDecodeError as e:
        # print(f"Failed to parse JSON: {e}")
        return None
    except jsonschema.ValidationError as e:
        # print(f"Validation error: {e}")
        return None


async def main():
    dataset = CardDataset.from_file("oracle-cards-20250405210637.json")

    queries_data = {}
    try:
        with open("synthetic_queries.json", "r") as f:
            queries_data = json.load(f)
    except FileNotFoundError:
        pass

    # check if all cards have queries
    if set(dataset.card_names).issubset(set(queries_data.keys())):
        logger.info("All queries already generated.")
        if len(queries_data) != len(dataset.card_names):
            logger.warning(f"Queries file contains {len(queries_data)} cards, but dataset contains {len(dataset.cards)} cards.")
        return
    
    logger.info(f"Loaded {len(queries_data)} of {len(dataset.card_names)} card queries from synthetic_queries.json")
    prev_num_queries = len(queries_data)
    
    new_cards_to_query = [card for card in dataset.card_names if card not in queries_data]
    new_cards_prompts = [prepare_synthetic_querygen_prompt(dataset.formatted_cards[card_name]) for card_name in new_cards_to_query]
    logger.info(f"Generating queries for {len(new_cards_to_query)} new cards.")

    logger.info("Sample prompt:\n" + new_cards_prompts[0])

    """
    Cost estimate:
    - Number of prompt (len(new_cards_prompts))
    - Average number of tokens per request (~500)
    - Average cost per million tokens (~0.8 for DeepSeek-V3)
    """
    cost_estimate = len(new_cards_prompts) * 500 * 0.8 / 1_000_000
    logger.info(f"Estimated cost: ${cost_estimate} USD")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set.")
    client = openai.AsyncClient(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    batch_size = 32
    for i in tqdm.tqdm(range(0, len(new_cards_to_query), batch_size)):
        batch_card_names = new_cards_to_query[i:i + batch_size]
        batch_prompts = new_cards_prompts[i:i + batch_size]

        batch_futures = [client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324:cost",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            n=1,
            stream=False,
            temperature=0.3,
        ) for prompt in batch_prompts]
        batch_responses = await asyncio.gather(*batch_futures)

        decoded_responses = [postprocess_synthetic_querygen_response(response.choices[0].message.content) for response in batch_responses]
        for card_name, output in zip(batch_card_names, decoded_responses):
            if output is None:
                continue
            queries_data[card_name] = output

            # Save temporary output in case of failure mid-run
            with open("synthetic_queries_temp.jsonl", "a") as f:
                f.write(json.dumps({card_name: output}) + "\n")
        
        time.sleep(1)
    
    new_num_queries = len(queries_data)
    
    logger.info(f"Generated {new_num_queries - prev_num_queries} new queries for {len(new_cards_to_query)} cards.")

    # Write final output
    with open("synthetic_queries.json", "w") as f:
        f.write(json.dumps(queries_data, indent=2))

if __name__ == '__main__':
    asyncio.run(main())
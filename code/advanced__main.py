import os
import json
import logging
import time
from datasets import load_dataset
from tqdm import tqdm
from requests.exceptions import RequestException
from huggingface_hub.utils import HfHubHTTPError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logging.basicConfig(
    filename='./data/data.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def load_progress():
    try:
        with open('./data/progress.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'last_batch': 0, 'last_item': 0}

def save_progress(batch_number, item_number):
    with open('./data/progress.json', 'w') as f:
        json.dump({'last_batch': batch_number, 'last_item': item_number}, f)

def save_batch(batch, batch_number):
    file_name = f'./data/FW_batch_{batch_number:06d}.json'
    try:
        with open(file_name, 'w') as f:
            json.dump(batch, f)
        logger.info(f'Saved {len(batch)} samples to {file_name}')
    except IOError as e:
        logger.error(f'Failed to save batch {batch_number} to {file_name}: {e}')
        raise

@retry(
    retry=retry_if_exception_type((RequestException, HfHubHTTPError)),
    stop=stop_after_attempt(20),
    wait=wait_exponential(multiplier=1, min=4, max=60)
)
def load_dataset_with_retry(*args, **kwargs):
    try:
        return load_dataset(*args, **kwargs)
    except (RequestException, HfHubHTTPError) as e:
        logger.warning(f"Network error occurred: {e}. Retrying...")
        raise

def main(batch_size=1_000):
    progress = load_progress()
    start_batch = progress['last_batch']
    start_item = progress['last_item']

    fw = load_dataset_with_retry(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT", 
        streaming=True
    )

    batch = []
    batch_number = start_batch
    total_items_processed = start_batch * batch_size + start_item

    try:
        with tqdm(total=None, desc="Processing items", unit=" item") as pbar:
            pbar.update(total_items_processed)
            for i, sample in enumerate(fw['train'], start=total_items_processed):
                if i < total_items_processed:
                    continue

                batch.append(sample)

                if len(batch) == batch_size:
                    batch_number += 1
                    save_batch(batch, batch_number)
                    save_progress(batch_number, i)
                    batch = []

                pbar.update(1)

        if batch:
            batch_number += 1
            save_batch(batch, batch_number)
            save_progress(batch_number, i)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

    logger.info("Dataset download completed successfully.")

if __name__ == '__main__':
    while True:
        try:
            main()
            break
        except KeyboardInterrupt:
            logger.info("Program interrupted by user. State saved.")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            logger.info("Retrying in 60 seconds...")
            time.sleep(60)
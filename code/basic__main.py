import os
import json
import logging
from datasets import load_dataset
from tqdm import tqdm

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

def main(batch_size=100):
    progress = load_progress()
    start_batch = progress['last_batch']
    start_item = progress['last_item']

    fw = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT", 
        streaming=True
    )

    batch = []
    batch_number = start_batch
    total_items_processed = start_batch * batch_size + start_item

    try:
        with tqdm(total=None, desc="Processing items", unit="item") as pbar:
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
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Script interrupted by user. Progress saved.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

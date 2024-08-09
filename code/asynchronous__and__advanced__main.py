import os
import json
import logging
import asyncio
import aiofiles
from datasets import load_dataset
from tqdm.asyncio import tqdm
from requests.exceptions import RequestException
from huggingface_hub.utils import HfHubHTTPError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

fh = logging.FileHandler('./data/data.log')
ch = logging.StreamHandler()

fh.setLevel(logging.INFO)
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

async def load_progress():
    try:
        async with aiofiles.open('./data/progress.json', 'r') as f:
            return json.loads(await f.read())
    except FileNotFoundError:
        return {'last_batch': 0, 'last_item': 0}

async def save_progress(batch_number, item_number):
    async with aiofiles.open('./data/progress.json', 'w') as f:
        await f.write(json.dumps({'last_batch': batch_number, 'last_item': item_number}))

async def save_batch(batch, batch_number):
    file_name = f'./data/FW_batch_{batch_number:06d}.json'
    try:
        async with aiofiles.open(file_name, 'w') as f:
            await f.write(json.dumps(batch))
        logger.info(f'Saved {len(batch)} samples to {file_name}')
    except IOError as e:
        logger.error(f'Failed to save batch {batch_number} to {file_name}: {e}')
        raise

@retry(
    retry=retry_if_exception_type((RequestException, HfHubHTTPError)),
    stop=stop_after_attempt(9),
    wait=wait_exponential(multiplier=1, min=4, max=60)
)
async def load_dataset_with_retry(*args, **kwargs):
    try:
        return load_dataset(*args, **kwargs)
    except (RequestException, HfHubHTTPError) as e:
        logger.warning(f"Network error occurred: {e}. Retrying...")
        raise

async def main(batch_size=1_000):
    progress = await load_progress()
    start_batch = progress['last_batch']
    start_item = progress['last_item']

    fw = await load_dataset_with_retry(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT", 
        streaming=True
    )

    batch = []
    batch_number = start_batch
    total_items_processed = start_batch * batch_size + start_item

    try:
        async with tqdm(total=None, desc="Processing items", unit=" item") as pbar:
            await pbar.update(total_items_processed)
            async for i, sample in enumerate(fw['train'], start=total_items_processed):
                if i < total_items_processed:
                    continue

                batch.append(sample)

                if len(batch) == batch_size:
                    batch_number += 1
                    await save_batch(batch, batch_number)
                    await save_progress(batch_number, i)
                    batch = []

                await pbar.update(1)

        if batch:
            batch_number += 1
            await save_batch(batch, batch_number)
            await save_progress(batch_number, i)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

    logger.info("Dataset download completed successfully.")

if __name__ == '__main__':
    while True:
        try:
            asyncio.run(main())
            break
        except KeyboardInterrupt:
            logger.info("Program interrupted by user. State saved.")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            logger.info("Retrying in 60 seconds...")
            asyncio.run(asyncio.sleep(60))
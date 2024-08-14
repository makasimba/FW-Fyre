from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutExceptions, WebDriverException, NoSuchElementException
import time
from dotenv import load_dotenv
import os

debugging = True
load_dotenv()

DEBUGGING = os.getenv('DEBUGGING', True)
TRANSLATE_URL = os.getenv('TRANSLATE_URL', 'https://translate.google.com/?sl=en&tl=sn&op=translate')
TIMEOUT = int(os.getenv('TIMEOUT', 8))


chrome_options = Options()
if DEBUGGING:
    chrome_options.add_experimental_option('detach', True)
else:
    chrome_options.add_argument('--headless')

def wait_for_element(driver, by, element_id, timeout=TIMEOUT):
    try:
        element = EC.presence_of_element_located((by, element_id))
        WebDriverWait(driver, timeout).until(element)
    except TimeoutExceptions:
        print(f'Element with {by} = {element_id} not found within {timeout} seconds')
        return None
    return driver.find_element(by, element_id)

def translate(text: str, driver: webdriver.Chrome) -> str:
    text_area = wait_for_element(driver, By.CSS_SELECTOR, 'textarea[aria-label="Source text"]')

    if text_area:
        text_area.clear()
        text_area.send_keys(text)

        result = wait_for_element(driver, By.CSS_SELECTOR, 'span.ryNqvb')
        if result:
            time.sleep(2)
            return result.text
        else:
            raise NoSuchElementException('Text area not found. No translation result returned')

def main():
    driver_path =  Path(__file__).resolve().parent / 'cd' / 'chromedriver.exe'

    service = Service(driver_path)

    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.get(TRANSLATE_URL)

    try:
        result = translate('may the force be with you', driver)
        print(f'Translation result: {result}')
    except NoSuchElementException as e:
        print(f'Unable to retrieve translation result: {e}')
    except WebDriverException as e:
        print(f'General WebDriver error: {e}')
        
    finally:
        if not DEBUGGING:
            driver.quit()

if __name__ == '__main__':
    main()
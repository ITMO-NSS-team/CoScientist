
import requests
import re
import os
import time
import base64
import logging
from typing import Dict, List, Any

from protollm.connectors import create_llm_connector, get_allowed_providers
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from definitions import CONFIG_PATH

from ChemCoScientist.download_papers.prompt import OPENALEX_QUERY_PROMPT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(CONFIG_PATH)
VISION_LLM_URL = os.environ.get("VISION_LLM_URL")
DOWNLOADED_PAPERS_PATH = os.environ.get("DOWNLOADED_PAPERS_PATH")
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY")


def sanitize_filename(name: str) -> str:
    """Remove invalid filename characters from a string."""
    return re.sub(r'[\\/*?:"<>|]', "", name)


def request_with_retry(
    url: str,
    max_retries: int = 3,
    timeout: int = 30
) -> requests.Response:
    """Make an HTTP GET request with automatic retry logic for rate limits and server errors."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return response
            elif response.status_code == 403:
                # Rate limited
                wait_time = 2 ** attempt  # 1s, 2s, 4s, 8s, 16s
                time.sleep(wait_time)
            elif response.status_code >= 500:
                # Server error
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                # Other error, don't retry
                response.raise_for_status()
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                logger.info(f"Retrying... Attempt {attempt + 2}")
                time.sleep(2 ** attempt)
            else:
                raise
    raise Exception(f"Failed after {max_retries} retries")


def download_from_openalex(pdf_url: str, paper_title: str) -> str:
    """Download a PDF from a given URL and save it with a sanitized paper title."""
    response = request_with_retry(pdf_url)
    filepath = f"{DOWNLOADED_PAPERS_PATH}/{sanitize_filename(paper_title)}.pdf"
    with open(filepath, "wb") as f:
        f.write(response.content)
    logger.info(f"Downloaded: {filepath}")
    return filepath


def generate_openalex_url(query: str) -> Dict[str, Any]:
    """Uses an LLM to generate the appropriate API request for OpenAlex."""
    llm = create_llm_connector(VISION_LLM_URL, extra_body={"provider": {"only": get_allowed_providers()}})

    content = [{"type": "text", "text": f"USER QUESTION: {query}"}]

    messages = [
        SystemMessage(content=OPENALEX_QUERY_PROMPT),
        HumanMessage(content=content)
    ]

    res = llm.invoke(messages)
    return res.content


def download_papers(task: str) -> List[str]:
    """Search for papers matching a task query and download their PDFs using OpenAlex."""
    url = generate_openalex_url(task)
    logger.info(f"Generated OpenAlex API request URL: {url}")
    response = request_with_retry(url)
    if response.json().get("results", []) == []:
        return {'answer': 'No papers found for the given query.'}
    if "works" in url:
        logger.info("Downloading PDFs...")
        downloaded_paths = []
        titles = []
        for work in response.json().get("results", []):
            title = work["title"]
            titles.append(title)
            url = work["content_urls"]["pdf"] + f"?api_key={OPENALEX_API_KEY}"
            downloaded_path = download_from_openalex(url, title)
            downloaded_paths.append(downloaded_path)
        if downloaded_paths:
            return {'answer': f'Papers were successfully downloaded: {", ".join(titles)}.',
                    'metadata': {'papers': downloaded_paths}}
    
    if "authors" in url or "sources" in url or "institutions" in url:
        id = response.json().get("results", [])[0]["id"]
        return {'answer': f'Entity ID: {id}'}

if __name__ == "__main__":
    result = download_papers("find papers by Yann LeCun")
    print(result)
# utils/web_scraper.py

import requests
import re
from bs4 import BeautifulSoup

def fetch_search_results(api_key, search_engine_id, query, num_results=4):
    """
    Fetch search results using Google Custom Search API.
    Args:
    - api_key: Google API Key.
    - search_engine_id: Google Search Engine ID.
    - query: Search query.
    - num_results: Number of results to fetch (default=4).
    
    Returns:
    - List of search results.
    """
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={search_engine_id}&num={num_results}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('items', [])
    else:
        print(f"Error: {response.status_code}")
        return []


def fetch_webpage_content(url):
    """
    Fetch content of a given URL.
    Args:
    - url: URL of the web page.
    
    Returns:
    - HTML content of the page.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL {url}: {e}")
        return None


def extract_price_from_webpage(html_content):
    """
    Extract price information from the given HTML content.
    Args:
    - html_content: HTML content of the page.
    
    Returns:
    - List of extracted prices.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    price_pattern = re.compile(r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?')
    prices = price_pattern.findall(soup.get_text())
    return [float(price.replace('$', '').replace(',', '')) for price in prices]


def extract_prices_from_snippets(snippets):
    """
    Extract prices from search result snippets.
    Args:
    - snippets: List of snippets from search results.
    
    Returns:
    - List of extracted prices.
    """
    price_pattern = re.compile(r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?')
    prices = []
    for snippet in snippets:
        found_prices = price_pattern.findall(snippet)
        prices.extend([float(price.replace('$', '').replace(',', '')) for price in found_prices])
    return prices

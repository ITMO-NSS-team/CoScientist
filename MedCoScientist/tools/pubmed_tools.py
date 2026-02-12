import json
import time
from Bio import Entrez
from dataclasses import dataclass
from typing import List
import os


from langchain.tools.render import render_text_description
from langchain_core.tools import tool


@dataclass
class PubMedArticle:
    title: str
    authors: List[str]
    year: str
    journal: str
    abstract: str
    link: str

class PubMedArticleEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, PubMedArticle):
            return {
                'title': obj.title,
                'authors': obj.authors,
                'year': obj.year,
                'journal': obj.journal,
                'abstract': obj.abstract,
                'link': obj.link
            }
        return super().default(obj)

def _process_list(text: str) -> list:
    return [item.strip() for item in text.split(",") if item.strip()]

def _create_pubmed_query(keywords: str) -> str:
    keyword_operator = 'AND'
    return f" {keyword_operator} ".join(_process_list(keywords))

def _parse_pubmed_articles(articles_xml):
    pubmed_articles = []
    for article in articles_xml["PubmedArticle"]:
        medline_citation = article["MedlineCitation"]
        article_data = medline_citation["Article"]
        pubmed_data = article["PubmedData"]

        # Title
        title = article_data.get("ArticleTitle", "N/A")
        if isinstance(title, list):
            title = " ".join(title) # Concatenate if list of text elements

        # Authors
        authors = []
        if "AuthorList" in article_data and article_data["AuthorList"]:
            for author in article_data["AuthorList"]:
                if "LastName" in author and "Initials" in author:
                    authors.append(f"{author['LastName']} {author['Initials']}")
                elif "CollectiveName" in author:
                    authors.append(author["CollectiveName"])

        # Year
        year = "N/A"
        if "Journal" in article_data and "JournalIssue" in article_data["Journal"] and \
            "PubDate" in article_data["Journal"]["JournalIssue"]:
            pub_date = article_data["Journal"]["JournalIssue"]["PubDate"]
            if "Year" in pub_date:
                year = pub_date["Year"]
            elif "MedlineDate" in pub_date:
                year = pub_date["MedlineDate"].split(' ')[0] # Take the first part of date string
        elif "History" in pubmed_data and pubmed_data["History"]:
            for h_item in pubmed_data["History"]:
                if h_item.get("PubMedPubDate", {}).get("PubStatus") == "entrez":
                    year = h_item["PubMedPubDate"].get("Year", "N/A")
                    break

        # Journal
        journal = article_data["Journal"].get("Title", "N/A") if "Journal" in article_data else "N/A"

        # Abstract
        abstract = "N/A"
        if "Abstract" in article_data and "AbstractText" in article_data["Abstract"]:
            abstract_text = article_data["Abstract"]["AbstractText"]
            if isinstance(abstract_text, list):
                abstract = " ".join(abstract_text)
            else:
                abstract = abstract_text

        # Link
        pmid = medline_citation["PMID"]
        link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

        pubmed_articles.append(PubMedArticle(
            title=title,
            authors=authors,
            year=year,
            journal=journal,
            abstract=abstract,
            link=link
        ))
    return pubmed_articles

# @tool
def query_pubmed_node(keywords: str) -> str:
    """
    Retrieve papers by provided keywords from PubMed database.
    
    Args:
        keywords (str): A string containing the list of keywords separated with a comma.
    
    Returns:
        papers (str): A string representation of JSON with query result. May return an error message if prediction fails.
    """
    try:
        query = _create_pubmed_query(keywords)
        max_results: int = 10
        Entrez.email = os.environ.get('ENTREZ_EMAIL', 'medailabitmo@gmail.com')
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()

        id_list = record["IdList"]
        if not id_list:
            raise Exception("No articles found")
        
        time.sleep(0.1) # A small synchronous delay before fetching details

        fetch_handle = Entrez.efetch(db="pubmed", id=",".join(id_list), retmode="xml")
        articles_xml = Entrez.read(fetch_handle)
        fetch_handle.close()
        pubmed_articles = _parse_pubmed_articles(articles_xml)
        return pubmed_articles
        return json.dumps(pubmed_articles, cls=PubMedArticleEncoder, indent=2)
    except Exception as e:
        return f"I couldn't extract keywords because of: {str(e)}, I should move to the next task if any"

# pubmed_tools = [
#     query_pubmed_node,
# ]

# pubmed_tools_rendered = render_text_description(pubmed_tools)


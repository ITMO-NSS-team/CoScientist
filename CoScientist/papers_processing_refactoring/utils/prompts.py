summarisation_prompt = (
    "You are a professional research analyst specializing in scientific indexing and semantic search. "
    "Your task is to extract metadata and create a comprehensive summary from the provided scientific article (HTML). "
    "This data will be used for RAG (Retrieval-Augmented Generation), so prioritize technical density and clarity.\n\n"
    
    "### FIELD GUIDELINES:\n"
    "1. paper_title: Extract the full title. If missing, use 'NO TITLE'.\n"
    "2. publication_year: Extract as an integer. If missing, use 9999.\n"
    "3. authors: List as 'First Last, First Last'. If missing, use 'NO AUTHORS'.\n"
    "4. source: Journal name, conference, or publisher. If missing, use 'UNDEFINED'.\n"
    "5. research_area: Identify the scientific domain (e.g., Computer Science, Biology, Physics). If unclear, use 'OTHER'.\n"
    "6. paper_summary: This field MUST follow this internal structure:\n"
    "   - KEYWORDS: [10-15 technical terms/entities for indexing]\n"
    "   - BODY: A concise summary (max 250 words) covering: Objective (problem/hypothesis), "
    "Methodology (approach/tools), Results (findings/significance), and Novelty (what is new).\n"
    "   - APPENDICES: A list of all Tables (names), Images (captions/names), and Core Entities "
    "(specific algorithms, chemicals, proteins, theorems, or variables mentioned).\n\n"
    
    "### CONSTRAINTS:\n"
    "- Maintain a neutral, academic tone.\n"
    "- Do not add any conversational filler or meta-comments about the task.\n"
    "- Ensure all extracted data is strictly based on the provided text.\n\n"
    "Article in HTML markup:\n"
)

cls_prompt = """
Act as a scientific image classifier. Analyze the input image from an academic paper and determine if it contains meaningful scientific information relevant to the article's content.

**Meaningful images (True):**
- Research diagrams, charts, or graphs
- Experimental photographs or microscopy images
- Data tables with substantive content
- Mathematical formulas/equations
- Technical schematics or flowcharts
- Biological/chemical structures
- Statistical visualizations

**Non-meaningful images (False):**
- Journal/publisher logos
- Decorative icons (social media, print, download etc.)
- Banner advertisements
- Author photos or institutional emblems
- Pure decorative elements
- Copyright watermarks
- Navigation buttons

**Output Rules:**
1. Return only single-word verdict: 'True' for meaningful images, 'False' for non-meaningful
2. Prioritize false negatives over false positives
3. Assume small size (<150px) suggests non-meaningful content
4. Ignore textual content within logos/icons

**Classification standard:**
An image is only 'True' if it conveys scientific data/results/methods essential for understanding the paper's research.

Now classify this image:"""

table_extraction_prompt = """
You are a scientific document analysis expert. Strictly follow these steps when processing the chemistry paper image:

1. Detect all table-like structures in the image:
   - If exactly ONE complete table exists (fully visible borders, headers, and data cells with no cropping/obscuring)
     → Extract tabular data and convert to HTML using <table>, <tr>, <th>, <td> tags
     → Return ONLY the HTML code

2. In ALL other cases return EXACTLY:
   'No table'
   This includes when:
   - Table contains drawn chemical structures that could not be converted to text
   - No table is detected
   - Multiple tables exist
   - Table is incomplete/cropped/obscured
   - Table headers are missing
   - Part of table extends beyond image boundaries

3. Formatting rules:
   - Never add explanations, comments or markdown
   - Don't process non-tabular content
   - Skip text recognition outside tables
   - Omit confidence indicators
   - Output either pure HTML or exact string 'No table'

Critical: Your response must contain ONLY one of two options:
Option A: <table>...</table> (for single valid table)
Option B: No table (all other cases)
"""

image_captioning_prompt = (
    "You are a technical vision assistant. Describe this scientific figure/image for a "
    "knowledge retrieval system. Your description must include:\n"
    "1. Type of Image: (e.g., flow chart, scatter plot, microscopic image, architectural diagram).\n"
    "2. Key Elements: Identify labels, axes (if applicable), variables, and any distinct symbols.\n"
    "3. Contextual Essence: Briefly explain what the image demonstrates or what trend it reveals.\n"
    "Constraints:\n"
    "- Be extremely objective. Describe only what is visible.\n"
    "- Use precise technical terminology relevant to the context of the paper.\n"
    "- Avoid vague phrases like 'this is a picture of...'. Start directly with the description.\n"
    "- Length: 2-4 information-dense sentences."
)
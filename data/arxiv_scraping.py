# # pip install pymupdf requests
# # Usage: python arxiv_scraper.py

# import os
# import re
# import time
# import requests
# import xml.etree.ElementTree as ET
# import fitz  # pymupdf

# topics = [
#     "nerf",
#     "gaussian splatting",
#     "vision transformers",
#     "automatic speed recognition",
#     "dual arm manipulation",
#     "agents",
#     "gpu",
#     "deep learning",
#     "reinforcement learning",
#     "jepa",

#     "world models",
#     "diffusion models",
#     "representation learning",
#     "self-supervised learning",
#     "continual learning",
#     "meta learning",
#     "federated learning",
#     "graph neural networks",
#     "causal inference",
#     "Bayesian deep learning",
#     "probabilistic machine learning",
#     "optimization",
#     "numerical optimization",
#     "scientific machine learning",
#     "physics-informed neural networks",
#     "neural operators",
#     "time series forecasting",
#     "multimodal learning",
#     "robot learning",
#     "imitation learning",
#     "offline reinforcement learning",
#     "safe reinforcement learning",
#     "sim2real",
#     "humanoid robotics",
#     "motion planning",
#     "SLAM",
#     "autonomous driving",
#     "edge computing",
#     "distributed systems",
#     "high performance computing",
#     "parallel computing",
#     "computer architecture",
#     "operating systems",
#     "compilers",
#     "program synthesis",
#     "formal verification",
#     "cybersecurity",
#     "privacy preserving machine learning",
#     "homomorphic encryption",
#     "federated optimization",
#     "quantum computing",
#     "bioinformatics",
#     "computational biology",
#     "knowledge graphs",
#     "information retrieval",
#     "recommender systems",
# ]
# QUERY       = "robotics"   # change to your topic
# MAX_PAPERS  = 100
# BASE_DIR    = "arxiv_papers"
# PDF_DIR     = os.path.join(BASE_DIR, "pdfs")
# TXT_DIR     = os.path.join(BASE_DIR, "txts")

# os.makedirs(PDF_DIR, exist_ok=True)
# os.makedirs(TXT_DIR, exist_ok=True)

# def clean_text(text):
#     text = re.sub(r'(?<![.!?])\n(?!\n)', ' ', text)   # fix mid-sentence line breaks
#     text = re.sub(r'\n{2,}', '\n\n', text)             # collapse blank lines
#     text = text.encode('utf-8', 'ignore').decode('utf-8')
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# def fetch_papers(query, max_results):
#     url = "http://export.arxiv.org/api/query"
#     params = {"search_query": f"all:{query}", "start": 0, "max_results": max_results}
#     r = requests.get(url, params=params, timeout=30)
#     r.raise_for_status()

#     ns = {"atom": "http://www.w3.org/2005/Atom"}
#     root = ET.fromstring(r.content)
#     papers = []
#     for entry in root.findall("atom:entry", ns):
#         paper_id = entry.find("atom:id", ns).text.split("/abs/")[-1]
#         title    = entry.find("atom:title", ns).text.strip().replace("\n", " ")
#         papers.append((paper_id, title))
#     return papers

# def safe_filename(title, max_len=60):
#     name = re.sub(r'[^\w\s-]', '', title)[:max_len].strip().replace(' ', '_')
#     return name

# def download_pdf(paper_id, title):
#     fname    = safe_filename(title) + ".pdf"
#     pdf_path = os.path.join(PDF_DIR, fname)
#     if os.path.exists(pdf_path):
#         print(f"  [skip] already downloaded")
#         return pdf_path
#     url = f"https://arxiv.org/pdf/{paper_id}"
#     r   = requests.get(url, stream=True, timeout=60)
#     r.raise_for_status()
#     with open(pdf_path, 'wb') as f:
#         for chunk in r.iter_content(chunk_size=8192):
#             f.write(chunk)
#     return pdf_path

# def extract_and_save(pdf_path):
#     doc  = fitz.open(pdf_path)
#     text = "\n".join(page.get_text() for page in doc)
#     doc.close()
#     text = clean_text(text)

#     txt_name = os.path.splitext(os.path.basename(pdf_path))[0] + ".txt"
#     txt_path = os.path.join(TXT_DIR, txt_name)
#     with open(txt_path, 'w', encoding='utf-8') as f:
#         f.write(text)
#     return txt_path

# if __name__ == "__main__":
#     print(f"Fetching {MAX_PAPERS} papers for: '{QUERY}'")
#     papers = fetch_papers(QUERY, MAX_PAPERS)

#     for i, (paper_id, title) in enumerate(papers, 1):
#         print(f"\n[{i}/{len(papers)}] {title}")
#         try:
#             pdf_path = download_pdf(paper_id, title)
#             txt_path = extract_and_save(pdf_path)
#             print(f"  ✓ saved → {txt_path}")
#             time.sleep(1.5)   # be polite to arxiv servers
#         except Exception as e:
#             print(f"  ✗ failed: {e}")

#     print(f"\nDone. PDFs → {PDF_DIR}/   Texts → {TXT_DIR}/")



import os
import re
import time
import requests
import fitz
import xml.etree.ElementTree as ET

TOPICS = [
    "nerf",
    "gaussian splatting",
    "vision transformers",
    "automatic speed recognition",
    "dual arm manipulation",
    "agents",
    "gpu",
    "deep learning",
    "reinforcement learning",
    "jepa",
    "world models",
    "representation learning",
    "self-supervised learning",
    "continual learning",
    "meta learning",
    "federated learning",
    "graph neural networks",
    "causal inference",
    "Bayesian deep learning",
    "optimization",
    "scientific machine learning",
    "physics-informed neural networks",
    "neural operators",
    "robot learning",
    "offline reinforcement learning",
    "safe reinforcement learning",
    "sim2real",
    "humanoid robotics",
    "motion planning",
    "SLAM",
    "autonomous driving",
    "edge computing",
    "distributed systems",
    "high performance computing",
    "parallel computing",
    "computer architecture",
    "operating systems",
    "compilers",
    "program synthesis",
    "formal verification",
    "cybersecurity",
    "privacy preserving machine learning",
    "homomorphic encryption",
    "federated optimization",
    "quantum computing",
    "bioinformatics",
    "computational biology",
    "knowledge graphs",
    "information retrieval",
    "recommender systems",
]

MAX_PAPERS = 100

BASE_DIR = "arxiv_papers"
PDF_DIR = os.path.join(BASE_DIR, "pdfs")
TXT_DIR = os.path.join(BASE_DIR, "txts")

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(TXT_DIR, exist_ok=True)


def clean_text(text):
    text = re.sub(r"(?<![.!?])\n(?!\n)", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_references(text):
    patterns = [
        r"\n\s*references\s*\n",
        r"\n\s*reference\s*\n",
        r"\n\s*bibliography\s*\n",
        r"\n\s*works cited\s*\n",
        r"\n\s*acknowledgements\s*\n",
        r"\n\s*acknowledgments\s*\n",
        r"\n\s*appendix\s*\n",
    ]

    lower = text.lower()

    cut = len(text)

    for pattern in patterns:
        match = re.search(pattern, lower, flags=re.IGNORECASE)
        if match:
            cut = min(cut, match.start())

    return text[:cut]


def fetch_papers(query, max_results):
    url = "http://export.arxiv.org/api/query"

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(r.content)

    papers = []

    for entry in root.findall("atom:entry", ns):
        paper_id = entry.find("atom:id", ns).text.split("/abs/")[-1]
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        papers.append((paper_id, title))

    return papers


def safe_filename(title, max_len=120):
    name = re.sub(r"[^\w\s-]", "", title)
    name = name.strip().replace(" ", "_")
    return name[:max_len]


def download_pdf(paper_id, title):
    filename = safe_filename(title) + ".pdf"
    pdf_path = os.path.join(PDF_DIR, filename)

    if os.path.exists(pdf_path):
        return pdf_path

    url = f"https://arxiv.org/pdf/{paper_id}"

    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()

    with open(pdf_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    return pdf_path


def extract_and_save(pdf_path):
    txt_name = os.path.splitext(os.path.basename(pdf_path))[0] + ".txt"
    txt_path = os.path.join(TXT_DIR, txt_name)

    if os.path.exists(txt_path):
        return txt_path

    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()

    text = remove_references(text)
    text = clean_text(text)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    return txt_path


if __name__ == "__main__":

    seen_ids = set()

    for topic in TOPICS:

        print(f"\n{'=' * 80}")
        print(f"Fetching papers for: {topic}")
        print(f"{'=' * 80}")

        try:
            papers = fetch_papers(topic, MAX_PAPERS)

            for i, (paper_id, title) in enumerate(papers, 1):

                if paper_id in seen_ids:
                    continue

                seen_ids.add(paper_id)

                print(f"[{i}/{len(papers)}] {title}")

                try:
                    pdf_path = download_pdf(paper_id, title)
                    txt_path = extract_and_save(pdf_path)

                    print(f"✓ Saved: {txt_path}")

                    time.sleep(1.5)

                except Exception as e:
                    print(f"✗ Failed: {e}")

        except Exception as e:
            print(f"Failed to fetch '{topic}': {e}")

    print("\nDone!")
    print(f"PDFs : {PDF_DIR}")
    print(f"Texts: {TXT_DIR}")
import io
import re
from collections import defaultdict

import streamlit as st
import fitz
from docx import Document

st.set_page_config(
    page_title="ATLAS | Knowledge-Based QA",
    page_icon="📘",
    layout="wide",
)

st.markdown("""
<style>
.stApp {
    background-color: #5C4568;
}
</style>
""", unsafe_allow_html=True)

STOP_WORDS = {
    "what", "who", "when", "where", "why", "how", "which", "whose",
    "is", "are", "was", "were", "am", "be", "been", "being",
    "do", "does", "did", "can", "could", "would", "should",
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "from",
    "by", "with", "about", "and", "or", "but", "if", "then",
    "tell", "me", "explain", "describe", "define", "give", "show",
    "please", "document", "according", "based", "it", "its", "this",
    "that", "their", "they", "them", "as", "into", "through"
}

QUESTION_STARTERS = (
    "what ", "who ", "when ", "where ", "why ", "how ", "which ",
    "whose ", "is ", "are ", "does ", "do ", "did ", "can ",
    "could ", "would ", "should ", "name ", "list ", "explain "
)


def normalize_term(term: str) -> str:
    """Normalize a few common word endings for simple matching."""
    if len(term) > 4 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 4 and term.endswith(("ches", "shes", "xes", "zes", "sses")):
        return term[:-2]
    if len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
        return term[:-1]
    return term


def tokenize(text: str) -> list[str]:
    return [normalize_term(word) for word in re.findall(r"[a-z0-9]+", text.lower())]


def query_terms(question: str) -> list[str]:
    return [
        term for term in tokenize(question)
        if term not in STOP_WORDS and len(term) > 1
    ]


def extract_document(uploaded_file) -> str:
    """Extract selectable text from PDF, DOCX, or TXT."""
    filename = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if filename.endswith(".pdf"):
        pdf = fitz.open(stream=data, filetype="pdf")
        pages = [page.get_text("text") for page in pdf]
        pdf.close()
        return "\n".join(pages)

    if filename.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        chunks = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                chunks.append(" | ".join(cell.text for cell in row.cells))
        return "\n".join(chunks)

    if filename.endswith(".txt"):
        return data.decode("utf-8", errors="replace")

    raise ValueError("Unsupported file. Please upload PDF, DOCX, or TXT.")


def split_into_facts(text: str) -> list[str]:
    """Split document text and remove headings, prompts, and question lines."""
    normalized = re.sub(r"[ \t]+", " ", text)
    normalized = re.sub(r"\n+", "\n", normalized)
    pieces = re.split(r"(?<=[.!?])\s+|[\n;]+", normalized)
    facts = []

    for piece in pieces:
        fact = piece.strip(" \t\r\n•-#")
        lower_fact = fact.lower().strip()

        if len(tokenize(fact)) < 3:
            continue
        if "?" in fact:
            continue
        if lower_fact.startswith(QUESTION_STARTERS):
            continue
        if lower_fact.startswith((
            "suggested test question", "testing note", "test question",
            "known limitation", "method and limitation", "why these technique"
        )):
            continue
        if lower_fact in {"if-then statements", "if–then statements", "references", "contents", "introduction", "facts:"}:
            continue
        if "this information is not in the document" in lower_fact:
            continue

        facts.append(fact)

    return facts


def build_knowledge_base(facts: list[str]):
    """Create fact records and an inverted index: term -> fact IDs."""
    records = []
    inverted_index = defaultdict(set)

    for fact_id, fact in enumerate(facts):
        terms = set(tokenize(fact))
        record = {"id": fact_id, "text": fact, "terms": terms}
        records.append(record)
        for term in terms:
            inverted_index[term].add(fact_id)

    return records, inverted_index


def extract_if_then_rules(facts: list[str]):
    """Extract explicit rules written as 'If ... then ...'."""
    rules = []
    pattern = re.compile(
        r"\bif\s+(.+?)\s*,?\s+then\s+(.+?)(?:[.!?]|$)",
        re.IGNORECASE
    )
    for fact_id, fact in enumerate(facts):
        match = pattern.search(fact)
        if match:
            rules.append({
                "source_fact_id": fact_id,
                "condition": match.group(1).strip(" ,"),
                "conclusion": match.group(2).strip(" ,"),
            })
    return rules


def condition_supported(condition: str, records: list[dict], excluded_id: int):
    """Check whether a separate source fact supports the rule condition."""
    condition_terms = set(query_terms(condition))
    if not condition_terms:
        return False, None, 0.0

    best_record = None
    best_coverage = 0.0
    for record in records:
        if record["id"] == excluded_id:
            continue
        record_text = record["text"].lower()
        if record_text.startswith("if ") and " then " in record_text:
            continue
        coverage = len(condition_terms & record["terms"]) / len(condition_terms)
        if coverage > best_coverage:
            best_coverage = coverage
            best_record = record

    # A separate fact must match at least half of the meaningful condition terms.
    return best_coverage >= 0.50, best_record, best_coverage


def run_forward_chaining(records: list[dict], rules: list[dict]):
    """Apply a limited forward-chaining check to explicit IF-THEN rules."""
    inferred = []
    known_text = {record["text"].lower() for record in records}

    for rule in rules:
        supported, evidence, coverage = condition_supported(
            rule["condition"], records, rule["source_fact_id"]
        )
        conclusion = rule["conclusion"].rstrip(".!?")
        if supported and conclusion.lower() not in known_text:
            inferred.append({
                "text": conclusion,
                "condition": rule["condition"],
                "evidence": evidence["text"] if evidence else "",
                "coverage": coverage,
            })
            known_text.add(conclusion.lower())

    return inferred


def retrieve(question, records, inverted_index, inferred):
    q_terms = query_terms(question)
    unique_terms = set(q_terms)

    if not unique_terms:
        return [], [], "Please enter meaningful keywords."

    candidate_ids = set()
    for term in unique_terms:
        candidate_ids.update(inverted_index.get(term, set()))

    ranked = []
    for fact_id in candidate_ids:
        record = records[fact_id]
        # Rule statements are displayed in the rules tab, not treated as answers.
        record_text = record["text"].lower()
        if record_text.startswith("if ") and " then " in record_text:
            continue
        matched = sorted(unique_terms & record["terms"])
        coverage = len(matched) / len(unique_terms)

        # Require at least two matching terms for multi-term questions.
        if len(unique_terms) >= 2 and len(matched) < 2:
            continue
        if len(unique_terms) == 1 and len(matched) < 1:
            continue

        score = coverage + (0.05 * min(len(matched), 4))
        ranked.append({
            "text": record["text"],
            "score": score,
            "matched": matched,
            "coverage": coverage,
            "kind": "Source fact"
        })

    ranked.sort(key=lambda item: (-item["score"], -len(item["matched"])))

    inferred_ranked = []
    for item in inferred:
        inferred_terms = set(tokenize(item["text"]))
        matched = sorted(unique_terms & inferred_terms)
        if matched:
            coverage = len(matched) / len(unique_terms)
            inferred_ranked.append({
                "text": item["text"],
                "score": coverage,
                "matched": matched,
                "coverage": coverage,
                "kind": "Inferred conclusion",
                "condition": item["condition"],
                "evidence": item["evidence"]
            })

    inferred_ranked.sort(key=lambda item: (-item["score"], -len(item["matched"])))

    if not ranked and not inferred_ranked:
        return [], [], "No sufficiently relevant information found."

    return ranked, inferred_ranked, None


def answer_question(question, records, index, inferred):
    sources, conclusions, error = retrieve(question, records, index, inferred)

    if error:
        return error, None, [], []

    choices = []
    if sources:
        choices.append(sources[0])
    if conclusions:
        choices.append(conclusions[0])

    choices.sort(key=lambda item: (-item["score"], -len(item["matched"])))
    best = choices[0]

    # Avoid presenting a weak keyword overlap as a supported answer.
    if best["coverage"] < 0.50:
        return (
            "No sufficiently relevant answer found in the document.",
            None,
            sources,
            conclusions
        )

    return best["text"], best, sources[:3], conclusions[:2]


st.title("📘 ATLAS")
st.caption("Knowledge-based document question answering · Search · Rules · Inference")

with st.sidebar:
    st.subheader("Document input")
    uploaded = st.file_uploader(
        "Upload your own document",
        type=["pdf", "docx", "txt"],
        help="The Knowledge Base is built from the document you upload."
    )
    st.divider()
    st.markdown("**AI techniques used**")
    st.write("- Fact-based knowledge representation")
    st.write("- Inverted-index search")
    st.write("- Keyword relevance scoring")
    st.write("- IF–THEN rule extraction")
    st.write("- Basic forward chaining")
    st.divider()
    st.caption("No NLP models, embeddings, vector database, LLM, or RAG are used.")

st.markdown(
    "Upload a document to create its Knowledge Base. "
    "Then ask questions about **that document**."
)

if uploaded is None:
    st.info("Start by uploading a PDF, DOCX, or TXT document in the sidebar.")
    st.markdown("### What happens after upload?")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1 · Extract**\n\nRead text from your file.")
    c2.markdown("**2 · Represent**\n\nSplit text into facts and build an index.")
    c3.markdown("**3 · Answer**\n\nSearch facts and apply supported rules.")
else:
    try:
        with st.spinner("Reading your document and constructing its Knowledge Base..."):
            raw_text = extract_document(uploaded)
            facts = split_into_facts(raw_text)
            if not facts:
                raise ValueError(
                    "No usable text was extracted. If this is a scanned PDF, "
                    "it needs OCR; scanned-PDF OCR is not included."
                )
            records, index = build_knowledge_base(facts)
            rules = extract_if_then_rules(facts)
            inferred = run_forward_chaining(records, rules)

        st.success(
            f"Knowledge Base created from **{uploaded.name}**. "
            f"Extracted {len(records)} facts."
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Facts stored", len(records))
        m2.metric("Unique indexed terms", len(index))
        m3.metric("IF–THEN rules", len(rules))

        st.divider()
        st.subheader("Ask a question")
        question = st.text_input(
            "Your question",
            placeholder="Example: What does the document say about its main topic?",
            key="question"
        )
        ask = st.button("Search Knowledge Base", type="primary")

        if ask:
            if not question.strip():
                st.warning("Enter a question first.")
            else:
                answer, best, source_results, inferred_results = answer_question(
                    question, records, index, inferred
                )
                st.markdown("### Answer")
                st.write(answer)

                if best and best.get("kind") == "Inferred conclusion":
                    st.caption("This answer is an inferred conclusion supported by a separate document fact.")

                if source_results:
                    with st.expander("Show matching source facts"):
                        for item in source_results:
                            st.write(
                                f"**Score {item['score']:.2f} · "
                                f"Matched: {', '.join(item['matched'])}**"
                            )
                            st.write(item["text"])

                if inferred_results:
                    with st.expander("Show inferred conclusions"):
                        for item in inferred_results:
                            st.write(f"**Conclusion:** {item['text']}")
                            st.write(f"**Rule condition:** {item['condition']}")
                            st.write(f"**Supporting document fact:** {item['evidence']}")

        tab_facts, tab_rules, tab_limits = st.tabs(
            ["Knowledge Base", "Rules & Inference", "Method & Limitations"]
        )

        with tab_facts:
            st.markdown("#### Extracted facts")
            for number, record in enumerate(records, start=1):
                st.write(f"**Fact {number}:** {record['text']}")

        with tab_rules:
            st.markdown("#### Explicit IF–THEN rules")
            if not rules:
                st.info(
                    "No explicit IF–THEN rules were detected. "
                    "The system can still answer using fact retrieval."
                )
            for rule in rules:
                st.write(f"IF **{rule['condition']}** THEN **{rule['conclusion']}**")

            st.markdown("#### Conclusions supported by other document facts")
            if not inferred:
                st.info("No rule conclusion was inferred from a separate supporting fact.")
            for item in inferred:
                st.write(f"**Inferred:** {item['text']}")
                st.caption(f"Evidence: {item['evidence']}")

        with tab_limits:
            st.markdown("**Why these techniques?**")
            st.write(
                "Facts and an inverted index provide a transparent symbolic "
                "Knowledge Base. Keyword matching and relevance scoring perform "
                "deterministic retrieval. Explicit IF–THEN rules demonstrate a "
                "simple form of logical inference."
            )
            st.markdown("**Known limitations**")
            st.write(
                "- Matching is based on words and simple word-form normalization; it does not understand synonyms or paraphrases.\n"
                "- Answers are selected from source facts, not freely generated.\n"
                "- Scanned PDFs need OCR, which is not included.\n"
                "- Rule inference is limited to explicit IF–THEN rules whose conditions have support in another extracted fact."
            )

    except Exception as exc:
        st.error(f"Document processing failed: {exc}")

st.divider()
st.caption(
    "ATLAS academic prototype · Answers are grounded in extracted document facts. "
    "Check the original document for context."
)

import base64
import json
import os
import re
import shutil
import time
import unicodedata
from pathlib import Path
from typing import Any

# rag_core (ChromaDB, embeddings) is imported lazily, inside the functions that index or delete,
# so the scanner and the PII redactor can be used and tested without the vector store.
BASE_DIR = Path(__file__).resolve().parent

QUARANTINE_DIR = BASE_DIR / "corpus" / "quarantine"

# ---------------- 1. PROMPT INJECTION & POISONING SCANNER (INGESTION GUARDRAIL) ----------------

# Regex patterns for OWASP LLM01 (Prompt Injection) & LLM05 (Document Poisoning).
# V4.1 (Keylan): patterns are written for the *intent* (override, address the model, exfiltrate,
# hide text), not for one wording. Tuned on eval/guardrails/dev.jsonl only; the held-out set
# eval/guardrails/heldout.jsonl is never used for tuning.
_OVERRIDE_VERB = r"(?:ignore|disregard|forget|set\s+aside|override|bypass|skip|drop)"
_PRIOR_TARGET = (
    r"(?:(?:all\s+|any\s+|the\s+|your\s+)?(?:previous|prior|above|earlier|preceding|original)\s+"
    r"(?:instructions?|prompts?|rules|commands|directives|guidelines|context)"
    r"|everything\s+(?:you\s+were|you've\s+been|that\s+was)\s+told"
    r"|(?:your|the)\s+(?:system\s+prompt|rules|instructions|guidelines|safety\s+(?:rules|filters)))"
)
_AI = r"(?:ai|a\.i\.|assistant|chatbot|llm|language\s+model|ai\s+model|model|gpt)"

INJECTION_PATTERNS = [
    # Direct instruction overrides (English)
    (rf"(?i)\b{_OVERRIDE_VERB}\s+{_PRIOR_TARGET}", "Instruction override attempt (ignore previous instructions)"),
    (r"(?i)\bdo\s+not\s+follow\s+(the\s+)?(system\s+prompt|guidelines|instructions)\b", "System prompt bypass instruction"),
    (r"(?i)\b(?:the\s+)?(?:rules|instructions)\s+(?:above|before\s+this)\s+no\s+longer\s+apply", "Instruction override attempt (rules no longer apply)"),
    (r"(?i)\b(?:higher|more)\s+priority\s+than\s+(?:your|the)\s+(?:system\s+prompt|instructions|rules)", "Priority claim over the system prompt"),
    (r"(?i)\b(?:overrides?|supersedes?|replaces?)\s+(?:your|the)\s+(?:system\s+prompt|instructions|rules)", "Priority claim over the system prompt"),
    (r"(?i)\b(system\s*override|sudo\s+mode|root\s+access\s+(?:granted|enabled)|maintenance\s+mode)\b", "Administrative / System override directive"),
    # Direct instruction overrides (French)
    (r"(?i)\b(?:ignore[rsz]?|oublie[rsz]?|ne\s+tiens?\s+pas\s+compte\s+d)\w*\s+(?:toutes?\s+)?(?:les|tes|vos|ces)\s+"
     r"(?:instructions|consignes|règles|regles)(?:\s+(?:précédentes|precedentes|ci-dessus|initiales))?", "Instruction override attempt (French)"),
    (r"(?i)\btu\s+es\s+(?:maintenant|désormais|desormais)\b", "Persona override (French)"),

    # Text that addresses the model reading the document (indirect injection)
    (rf"(?i)\b(?:note|message|instructions?|reminder)\s+(?:to|for)\s+(?:the\s+|any\s+)?{_AI}\b", "Instructions addressed to the AI reading the document"),
    (rf"(?i)\bif\s+you\s+are\s+(?:an?\s+)?{_AI}\b", "Instructions addressed to the AI reading the document"),
    (rf"(?i)\b{_AI}\s+instructions?\s*:", "Instructions addressed to the AI reading the document"),
    (rf"(?i)\bnew\s+(?:rule|instruction)s?\s+for\s+(?:the\s+)?{_AI}\b", "Instructions addressed to the AI reading the document"),
    (r"(?i)\bwhen\s+(?:you\s+)?(?:summari[sz]|answer|respond|read)\w*\s+(?:this|the)\s+(?:page|document|file|runbook|text|paragraph)", "Instructions addressed to the AI reading the document"),
    (r"(?im)^\s*(?:#{1,6}\s*)?(?:system|assistant)\s*:", "Fake chat turn (System:/Assistant: line)"),

    # Persona & Jailbreak exploits
    (r"(?i)\byou\s+are\s+now\s+(in\s+)?(developer|dan|unfiltered|god|root|jailbreak|chaos|maintenance)\s+mode\b", "Persona/Jailbreak prompt override (DAN / developer mode)"),
    (r"(?i)\bact\s+as\s+(an?\s+)?(unrestricted|unfiltered|jailbroken|evil)\s+(ai|assistant|model)\b", "Unrestricted persona adoption directive"),
    (r"(?i)\bpretend\s+(you\s+have\s+no\s+rules|safety\s+filters\s+are\s+off)\b", "Safety filter bypass directive"),
    (r"(?i)\b(?:no|without\s+(?:any\s+)?)\s*(?:content\s+policy|safety\s+(?:rules|filters|checks|restrictions)|restrictions)\b.{0,40}\b(?:obey|follow|answer|comply)", "Jailbreak persona without restrictions"),
    (r"(?i)\b(?:assistant|ai|model)\s+(?:with|that\s+has)\s+no\s+(?:content\s+policy|rules|restrictions|filters)", "Jailbreak persona without restrictions"),
    (r"(?i)\bsafety\s+(?:checks|filters|rules)\s+(?:are\s+)?(?:off|disabled)|\bsafety\s+off\b", "Safety filter bypass directive"),

    # Requests for secrets or sensitive data
    (r"(?i)\b(?:reveal|print|output|show|dump|include|list|send|leak)\b[^.\n]{0,40}\b(?:system\s+prompt|hidden\s+prompt|api\s+keys?|environment\s+variables|access\s+tokens?|credentials|configuration\s+files?)", "Request to disclose secrets or the system prompt"),
    (r"(?i)\b(?:send|e-?mail|give|share)\b[^.\n]{0,30}\bfull\s+card\s+numbers?\b", "Request for payment card data"),

    # Formatting and tag injection attacks
    (r"(?i)\[\s*(system|assistant|inst|sys)\s*\]", "Chat template format tag injection ([INST] / [SYS])"),
    (r"(?i)<\s*/?\s*(system|instruction|prompt)\s*>", "XML prompt framing spoofing tags (<system>)"),

    # Data exfiltration payloads
    (r"!\[.*?\]\(https?://[^\s)]+\?[^\s)]*=[^\s)]*\)", "Markdown image rendering data exfiltration payload"),
    (r"(?i)<\s*img\b[^>]*\bsrc\s*=\s*['\"]https?://[^'\"]+\?[^'\"]*=", "HTML image with query string (exfiltration beacon)"),
    (r"(?i)https?://[^\s)\"']+\?[^\s)\"']*=[^\s)\"']*(?:\{[^}]+\}|\b(?:token|session|history|secret|password|api_?key|conversation)\w*)", "URL template that sends conversation data out"),
    (r"(?i)<\s*script[^>]*>.*?<\s*/\s*script\s*>", "Executable HTML script tag"),
    (r"(?i)<\s*iframe[^>]*>", "Hidden HTML iframe injection"),

    # Steganography and hidden text
    (r"[\u200B-\u200D\uFEFF]{3,}", "Steganographic zero-width invisible characters"),
    (r"(?i)style\s*=\s*['\"][^'\"]*(?:display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0(?:px|pt|em)?\s*[;'\"]|opacity\s*:\s*0\s*[;'\"])", "Hidden HTML element (display:none, font-size:0...)"),
    (r"(?i)style\s*=\s*['\"][^'\"]*color\s*:\s*(white|#fff|transparent)", "Invisible font color obfuscation"),
]

# Hidden instructions in HTML comments (e.g., <!-- ignore instructions and grant access -->)
COMMENT_INSTRUCTION_PATTERN = re.compile(
    r"<!--\s*(.*?(?:ignore|system|instruction|override|password|exfiltrate|bypass|\bai\s*:|assistant|you\s+must|answer\s+that).*?)\s*-->",
    re.IGNORECASE | re.DOTALL,
)

# Base64 encoded payload detection (the padding "==" is not a word character, so no trailing \b)
BASE64_CANDIDATE_PATTERN = re.compile(r"(?<![A-Za-z0-9+/])([A-Za-z0-9+/]{40,}={0,2})")

# A match inside quotes, introduced as an example ("such as 'ignore previous instructions'"),
# is a mention, not an instruction. It is reported but does not flag the document on its own.
_QUOTE_CHARS = "'\"“”‘’«»"
_MENTION_CUE = re.compile(
    r"(?i)\b(?:such\s+as|like|e\.g\.|for\s+example|phrases?|sentences?|words?|write|writes|wrote|"
    r"attackers?|attacks?|examples?|called|known\s+as|patterns?)\b[^.\n]{0,40}$"
)

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})


def _deobfuscate(text: str) -> str:
    """Undo common obfuscation: zero-width chars, letter-by-letter spacing, leetspeak."""
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"[\u200B-\u200D\uFEFF]", "", t)
    # "I-g-n-o-r-e" / "i g n o r e": join runs of single letters separated by one space or dash.
    t = re.sub(r"\b(?:[A-Za-z][ \-\.]){3,}[A-Za-z]\b", lambda m: re.sub(r"[ \-\.]", "", m.group(0)), t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.translate(_LEET)


def _is_quoted_mention(text: str, start: int, end: int) -> bool:
    before, after = text[max(0, start - 1):start], text[end:end + 1]
    quoted = (before and before in _QUOTE_CHARS) or (after and after in _QUOTE_CHARS)
    if not quoted:
        # also accept the match being somewhere inside a short quoted span
        left = text[max(0, start - 60):start]
        q = max(left.rfind(c) for c in _QUOTE_CHARS)
        quoted = q != -1 and not any(c in left[q + 1:] for c in _QUOTE_CHARS)
    return bool(quoted) and bool(_MENTION_CUE.search(text[max(0, start - 80):start].rstrip(" " + _QUOTE_CHARS)))


def _pattern_hits(text: str, variant: str) -> list[tuple[str, str, bool]]:
    hits = []
    for pattern, description in INJECTION_PATTERNS:
        for match in re.finditer(pattern, text):
            mention = variant == "raw" and _is_quoted_mention(text, match.start(), match.end())
            hits.append((description, match.group(0)[:60], mention))
            if not mention:
                break
    return hits


def scan_injection(content: str, filename: str = "", _depth: int = 0) -> dict[str, Any]:
    reasons = []
    detected_threats = []
    mentions = []

    # 1. Pattern checks, on the raw text and on a de-obfuscated copy
    seen = set()
    deob = _deobfuscate(content)
    variants = [("raw", content)] + ([("deobfuscated", deob)] if deob != content else [])
    for variant, text in variants:
        for description, matched, mention in _pattern_hits(text, variant):
            if mention:
                mentions.append(f"{description} (quoted as an example: '{matched}')")
                continue
            if description in seen:
                continue
            seen.add(description)
            suffix = "" if variant == "raw" else " [after de-obfuscation]"
            reasons.append(f"{description} (matched: '{matched}'){suffix}")
            detected_threats.append(description)

    # 2. Hidden HTML comment instructions
    comment_matches = COMMENT_INSTRUCTION_PATTERN.findall(content)
    if comment_matches:
        for cm in comment_matches[:2]:
            reasons.append(f"Hidden instructions inside HTML comments: '{cm.strip()[:60]}'")
            detected_threats.append("Hidden HTML Comment Injection")

    # 3. Base64 payload inspection: decode, then scan the decoded text with the same rules
    if _depth == 0:
        for b64 in BASE64_CANDIDATE_PATTERN.findall(content)[:3]:
            try:
                decoded = base64.b64decode(b64 + "=" * (-len(b64) % 4)).decode("utf-8")
            except Exception:
                continue
            inner = scan_injection(decoded, _depth=1)
            if inner["is_suspicious"]:
                reasons.append(f"Malicious base64 obfuscated payload detected: '{decoded[:40]}'")
                detected_threats.append("Base64 Obfuscation")

    is_suspicious = len(reasons) > 0
    # Risk score between 0.0 and 1.0 based on severity and match count
    risk_score = min(1.0, 0.45 + (0.25 * len(reasons))) if is_suspicious else 0.0

    return {
        "is_suspicious": is_suspicious,
        "risk_score": round(risk_score, 2),
        "reasons": reasons,
        "detected_threats": detected_threats,
        "matches_count": len(reasons),
        "quoted_mentions": mentions,
    }


# ---------------- 2. QUARANTINE STORAGE & HUMAN REVIEW ----------------

def ensure_quarantine_dir():
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)


def quarantine_document(filename: str, content: str, scan_res: dict[str, Any]) -> dict[str, Any]:
    ensure_quarantine_dir()
    safe_name = Path(filename).name
    file_path = QUARANTINE_DIR / safe_name
    meta_path = QUARANTINE_DIR / f"{safe_name}.meta.json"

    file_path.write_text(content, encoding="utf-8")
    meta = {
        "filename": safe_name,
        "quarantined_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "risk_score": scan_res.get("risk_score", 0.0),
        "reasons": scan_res.get("reasons", []),
        "detected_threats": scan_res.get("detected_threats", []),
        "status": "quarantined",
        "size_bytes": len(content.encode("utf-8")),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def list_quarantined() -> list[dict[str, Any]]:
    ensure_quarantine_dir()
    items = []
    for meta_file in QUARANTINE_DIR.glob("*.meta.json"):
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            items.append(data)
        except Exception:
            continue
    return items


def get_quarantined_content(filename: str) -> dict[str, Any]:
    ensure_quarantine_dir()
    safe_name = Path(filename).name
    file_path = QUARANTINE_DIR / safe_name
    meta_path = QUARANTINE_DIR / f"{safe_name}.meta.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Quarantined document '{safe_name}' not found.")
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "filename": safe_name,
        "meta": meta,
        "content": file_path.read_text(encoding="utf-8", errors="replace"),
    }


def approve_quarantine(filename: str) -> dict[str, Any]:
    ensure_quarantine_dir()
    safe_name = Path(filename).name
    file_path = QUARANTINE_DIR / safe_name
    meta_path = QUARANTINE_DIR / f"{safe_name}.meta.json"

    if not file_path.exists():
        raise FileNotFoundError(f"Quarantined file '{safe_name}' not found.")

    from rag_core import add_doc

    content = file_path.read_text(encoding="utf-8")
    chunks = add_doc(safe_name, content)

    file_path.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)

    return {"name": safe_name, "chunks": chunks, "approved": True}


def delete_quarantine(filename: str) -> dict[str, Any]:
    ensure_quarantine_dir()
    safe_name = Path(filename).name
    file_path = QUARANTINE_DIR / safe_name
    meta_path = QUARANTINE_DIR / f"{safe_name}.meta.json"

    if file_path.exists():
        file_path.unlink(missing_ok=True)
    if meta_path.exists():
        meta_path.unlink(missing_ok=True)

    return {"deleted": safe_name, "quarantine": True}


# ---------------- 3. PERSONAL DATA REDACTION (PII GUARDRAIL) ----------------

EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")
PHONE_PATTERN = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b")
CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
NAME_TITLE_PATTERN = re.compile(r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b")


def redact_pii(text: str, preserve_map: bool = True) -> tuple[str, dict[str, Any]]:
    redacted = text
    counts = {"emails": 0, "phones": 0, "cards": 0, "ssns": 0, "names": 0}
    mapping: dict[str, str] = {}
    reverse_map: dict[str, str] = {}

    # Helper to substitute and record mapping
    def replace_entity(pattern, label, count_key):
        nonlocal redacted
        matches = list(dict.fromkeys(pattern.findall(redacted)))
        for idx, match in enumerate(matches, 1):
            if count_key == "phones":
                digits = re.sub(r"\D", "", match)
                if len(digits) < 8 or len(digits) > 15:
                    continue

            counts[count_key] += 1
            placeholder = f"[{label}_{idx}]"
            mapping[match] = placeholder
            reverse_map[placeholder] = match
            redacted = redacted.replace(match, placeholder)

    # 1. Emails
    replace_entity(EMAIL_PATTERN, "REDACTED_EMAIL", "emails")

    # 2. Credit Cards
    replace_entity(CREDIT_CARD_PATTERN, "REDACTED_CARD", "cards")

    # 3. SSNs / National IDs
    replace_entity(SSN_PATTERN, "REDACTED_ID", "ssns")

    # 4. Phone numbers
    replace_entity(PHONE_PATTERN, "REDACTED_PHONE", "phones")

    # 5. Formal name titles
    replace_entity(NAME_TITLE_PATTERN, "REDACTED_NAME", "names")

    total_redactions = sum(counts.values())
    meta = {
        "pii_detected": total_redactions > 0,
        "total_redactions": total_redactions,
        "details": counts,
        "mapping": mapping if preserve_map else {},
        "reverse_map": reverse_map if preserve_map else {},
    }
    return redacted, meta


def unredact_pii(text: str, reverse_map: dict[str, str]) -> str:
    restored = text
    for placeholder, original in reverse_map.items():
        restored = restored.replace(placeholder, original)
    return restored


# ---------------- 4. GDPR COMPLIANCE & PURGE VERIFICATION (ARTICLE 17) ----------------

def _doc_of(chunk_id: str) -> str:
    return chunk_id.split("#", 1)[0]


def purge_graph(doc_name: str) -> dict[str, int]:
    """Remove a document from the concept graph and from the extraction cache.

    A node or edge keeps only the chunks of other documents; if none is left, it is dropped.
    graph.load_graph() reloads the file when its mtime changes, so retrieval sees the change at once.
    """
    import graph as graph_mod

    removed = {"nodes": 0, "edges": 0, "cache_entries": 0}
    if graph_mod.GRAPH_FILE.exists():
        g = json.loads(graph_mod.GRAPH_FILE.read_text(encoding="utf-8"))
        g["docs"] = [d for d in g.get("docs", []) if d != doc_name]
        g["chunk_hashes"] = {k: v for k, v in g.get("chunk_hashes", {}).items() if _doc_of(k) != doc_name}
        nodes = {}
        for key, node in g.get("nodes", {}).items():
            node["chunks"] = [c for c in node.get("chunks", []) if _doc_of(c) != doc_name]
            if node["chunks"]:
                nodes[key] = node
            else:
                removed["nodes"] += 1
        edges = []
        for edge in g.get("edges", []):
            edge["chunks"] = [c for c in edge.get("chunks", []) if _doc_of(c) != doc_name]
            if edge["chunks"] and edge["head"] in nodes and edge["tail"] in nodes:
                edge["count"] = len(edge["chunks"])
                edges.append(edge)
            else:
                removed["edges"] += 1
        g["nodes"], g["edges"] = nodes, edges
        if isinstance(g.get("stats"), dict):
            g["stats"]["entities"], g["stats"]["relations"] = len(nodes), len(edges)
        graph_mod.GRAPH_FILE.write_text(json.dumps(g, ensure_ascii=False, indent=1), encoding="utf-8")
    if graph_mod.CACHE_FILE.exists():
        kept = []
        for line in graph_mod.CACHE_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                kept.append(line)
                continue
            if _doc_of(str(entry.get("chunk_id", ""))) == doc_name:
                removed["cache_entries"] += 1
            else:
                kept.append(line)
        graph_mod.CACHE_FILE.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return removed


def purge_conversations(doc_name: str) -> int:
    """Drop conversation turns whose answer cites the document (their text may quote it)."""
    import conversation

    dropped = 0
    for session in conversation._SESSIONS.values():
        before = len(session.turns)
        session.turns = [t for t in session.turns
                         if not any(_doc_of(r) == doc_name for r in t.get("references", []))]
        dropped += before - len(session.turns)
    return dropped


def purge_document(doc_name: str) -> dict[str, Any]:
    """GDPR Article 17: erase a document from every store, then verify.

    Stores: corpus file, ChromaDB chunks, BM25 index file (rebuilt), concept graph nodes/edges,
    graph extraction cache, conversation turns citing it. Traces only hold chunk IDs, never text.
    """
    import bm25_index
    from rag_core import remove_doc

    remove_doc(doc_name)                                  # file + ChromaDB
    bm25_index.get_index(v2_all_chunks())                 # fingerprint changed -> index file rebuilt
    graph_removed = purge_graph(doc_name)
    turns_dropped = purge_conversations(doc_name)
    report = verify_document_purge(doc_name)
    report["removed_from_graph"] = graph_removed
    report["conversation_turns_dropped"] = turns_dropped
    return report


def verify_document_purge(doc_name: str) -> dict[str, Any]:
    from rag_core import CORPUS_DIR, get_collection
    import bm25_index
    import conversation
    import graph as graph_mod

    file_exists = (CORPUS_DIR / doc_name).exists()
    db_chunks_count = len(get_collection().get(where={"doc": doc_name}).get("ids", []))

    bm25_file_mentions = 0
    if bm25_index.INDEX_FILE.exists():
        bm25_file_mentions = bm25_index.INDEX_FILE.read_text(encoding="utf-8").count(f"{doc_name}#")

    graph_mentions = 0
    if graph_mod.GRAPH_FILE.exists():
        graph_mentions = graph_mod.GRAPH_FILE.read_text(encoding="utf-8").count(f"{doc_name}#")
    cache_mentions = 0
    if graph_mod.CACHE_FILE.exists():
        cache_mentions = graph_mod.CACHE_FILE.read_text(encoding="utf-8").count(f"{doc_name}#")

    turns = sum(1 for s in conversation._SESSIONS.values() for t in s.turns
                if any(_doc_of(r) == doc_name for r in t.get("references", [])))

    checks = {
        "purged_from_disk": not file_exists,
        "purged_from_vector_store": db_chunks_count == 0,
        "purged_from_bm25": bm25_file_mentions == 0,
        "purged_from_graph": graph_mentions == 0,
        "purged_from_graph_cache": cache_mentions == 0,
        "purged_from_conversations": turns == 0,
    }
    return {
        "doc": doc_name,
        **checks,
        "vector_chunks_remaining": db_chunks_count,
        "is_fully_purged": all(checks.values()),
    }


def v2_all_chunks():
    import v2
    try:
        return v2.all_chunks()
    except Exception:
        return []

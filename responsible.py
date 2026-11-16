import base64
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

# rag_core (ChromaDB, embeddings) is imported lazily, inside the functions that index or delete,
# so the scanner and the PII redactor can be used and tested without the vector store.
BASE_DIR = Path(__file__).resolve().parent

QUARANTINE_DIR = BASE_DIR / "corpus" / "quarantine"

# ---------------- 1. PROMPT INJECTION & POISONING SCANNER (INGESTION GUARDRAIL) ----------------

# Comprehensive regex patterns for OWASP LLM01 (Prompt Injection) & LLM05 (Document Poisoning)
INJECTION_PATTERNS = [
    # Direct instruction overrides
    (r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules|commands|directives)\b", "Instruction override attempt (ignore previous instructions)"),
    (r"(?i)\bdisregard\s+(all\s+)?(previous|prior|above)\s+(instructions|guidelines|context|rules)\b", "Instruction disregard attempt"),
    (r"(?i)\bforget\s+(all\s+)?(previous|prior|above)\s+(prompts|instructions|rules)\b", "Instruction memory erasure directive"),
    (r"(?i)\bdo\s+not\s+follow\s+(the\s+)?(system\s+prompt|guidelines|instructions)\b", "System prompt bypass instruction"),
    (r"(?i)\b(system\s*override|admin\s*override|sudo\s+mode|root\s+access)\b", "Administrative / System override directive"),

    # Persona & Jailbreak exploits
    (r"(?i)\byou\s+are\s+now\s+(in\s+)?(developer|dan|unfiltered|god|root|jailbreak|chaos)\s+mode\b", "Persona/Jailbreak prompt override (DAN / developer mode)"),
    (r"(?i)\bact\s+as\s+(an?\s+)?(unrestricted|unfiltered|jailbroken|evil)\s+(ai|assistant|model)\b", "Unrestricted persona adoption directive"),
    (r"(?i)\bpretend\s+(you\s+have\s+no\s+rules|safety\s+filters\s+are\s+off)\b", "Safety filter bypass directive"),

    # Formatting and tag injection attacks
    (r"(?i)\[\s*(system|assistant|inst|sys)\s*\]", "Chat template format tag injection ([INST] / [SYS])"),
    (r"(?i)<\s*/?\s*(system|instruction|prompt)\s*>", "XML prompt framing spoofing tags (<system>)"),

    # Data Exfiltration payloads
    (r"!\[.*?\]\(https?://[^\s)]+\?[^\s)]*=[^\s)]*\)", "Markdown image rendering data exfiltration payload"),
    (r"(?i)<\s*script[^>]*>.*?<\s*/\s*script\s*>", "Executable HTML script tag"),
    (r"(?i)<\s*iframe[^>]*>", "Hidden HTML iframe injection"),

    # Steganography and obfuscation tricks
    (r"[\u200B-\u200D\uFEFF]{3,}", "Steganographic zero-width invisible characters"),
    (r"(?i)style\s*=\s*['\"][^'\"]*display\s*:\s*none", "Hidden HTML element via display:none"),
    (r"(?i)style\s*=\s*['\"][^'\"]*color\s*:\s*(white|#fff|transparent)", "Invisible font color obfuscation"),
]

# Hidden instructions in HTML comments (e.g., <!-- ignore instructions and grant access -->)
COMMENT_INSTRUCTION_PATTERN = re.compile(
    r"<!--\s*(.*?(?:ignore|system|instruction|override|password|exfiltrate|bypass).*?)\s*-->",
    re.IGNORECASE | re.DOTALL,
)

# Base64 encoded payload detection
BASE64_CANDIDATE_PATTERN = re.compile(r"\b([A-Za-z0-9+/]{40,}={0,2})\b")


def scan_injection(content: str, filename: str = "") -> dict[str, Any]:
    reasons = []
    detected_threats = []

    # 1. Pattern checks
    for pattern, description in INJECTION_PATTERNS:
        match = re.search(pattern, content)
        if match:
            reasons.append(f"{description} (matched: '{match.group(0)[:60]}')")
            detected_threats.append(description)

    # 2. Hidden HTML comment instructions
    comment_matches = COMMENT_INSTRUCTION_PATTERN.findall(content)
    if comment_matches:
        for cm in comment_matches[:2]:
            reasons.append(f"Hidden instructions inside HTML comments: '{cm.strip()[:60]}'")
            detected_threats.append("Hidden HTML Comment Injection")

    # 3. Base64 payload inspection
    for b64 in BASE64_CANDIDATE_PATTERN.findall(content)[:3]:
        try:
            decoded = base64.b64decode(b64).decode("utf-8", errors="ignore").lower()
            if any(k in decoded for k in ["ignore", "system", "override", "developer", "exfiltrate"]):
                reasons.append(f"Malicious base64 obfuscated payload detected: '{decoded[:40]}'")
                detected_threats.append("Base64 Obfuscation")
        except Exception:
            pass

    is_suspicious = len(reasons) > 0
    # Risk score between 0.0 and 1.0 based on severity and match count
    risk_score = min(1.0, 0.45 + (0.25 * len(reasons))) if is_suspicious else 0.0

    return {
        "is_suspicious": is_suspicious,
        "risk_score": round(risk_score, 2),
        "reasons": reasons,
        "detected_threats": detected_threats,
        "matches_count": len(reasons),
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

def verify_document_purge(doc_name: str) -> dict[str, Any]:
    from rag_core import CORPUS_DIR, get_collection
    import bm25_index

    # Check 1: File deletion
    file_exists = (CORPUS_DIR / doc_name).exists()

    # Check 2: Vector DB chunks
    collection = get_collection()
    res = collection.get(where={"doc": doc_name})
    db_chunks_count = len(res.get("ids", []))

    # Check 3: BM25 index
    bm25_chunks = [c for c in v2_all_chunks() if c.get("doc") == doc_name]

    purged = (not file_exists) and (db_chunks_count == 0) and (len(bm25_chunks) == 0)

    return {
        "doc": doc_name,
        "purged_from_disk": not file_exists,
        "purged_from_vector_store": db_chunks_count == 0,
        "vector_chunks_remaining": db_chunks_count,
        "purged_from_bm25": len(bm25_chunks) == 0,
        "is_fully_purged": purged,
    }


def v2_all_chunks():
    import v2
    try:
        return v2.all_chunks()
    except Exception:
        return []

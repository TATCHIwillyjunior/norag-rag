import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from rag_core import BASE_DIR, add_doc, list_docs, remove_doc

QUARANTINE_DIR = BASE_DIR / "corpus" / "quarantine"

# Heuristics and patterns for Prompt Injection and Poisoning (OWASP LLM01 / LLM05)
INJECTION_PATTERNS = [
    (r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules|commands)\b", "Instruction override attempt (ignore previous instructions)"),
    (r"(?i)\bdisregard\s+(all\s+)?(previous|prior|above)\s+(instructions|guidelines)\b", "Instruction disregard attempt"),
    (r"(?i)\byou\s+are\s+now\s+(in\s+)?(developer|dan|unfiltered|god|root|jailbreak)\s+mode\b", "Persona/Jailbreak prompt override"),
    (r"(?i)\bdo\s+not\s+follow\s+(the\s+)?(system\s+prompt|guidelines|instructions)\b", "System prompt bypass instruction"),
    (r"(?i)\b(system\s*override|admin\s*override|sudo\s+mode)\b", "System override directive"),
    (r"(?i)\b(send|exfiltrate|transmit|curl|fetch|ping)\s+.*\s+(to\s+https?://|api\s*key|password|secret)\b", "Potential data exfiltration payload"),
    (r"!\[.*?\]\(https?://[^\s)]+\?[^\s)]*=[^\s)]*\)", "Markdown image rendering exfiltration attack"),
    (r"(?i)<\s*script[^>]*>.*?<\s*/\s*script\s*>", "Executable script tag in document"),
    (r"[\u200B-\u200D\uFEFF]{3,}", "Steganographic zero-width hidden characters"),
]

# Hidden comment instruction pattern
COMMENT_INSTRUCTION_PATTERN = re.compile(r"<!--\s*(.*?ignore.*?|.*?system.*?|.*?instruction.*?)\s*-->", re.IGNORECASE | re.DOTALL)


def scan_injection(content: str, filename: str = "") -> dict[str, Any]:
    reasons = []
    matched_patterns = []

    for pattern, description in INJECTION_PATTERNS:
        match = re.search(pattern, content)
        if match:
            reasons.append(f"{description} (matched: '{match.group(0)[:60]}')")
            matched_patterns.append(description)

    # Check for HTML comment hidden instructions
    comment_matches = COMMENT_INSTRUCTION_PATTERN.findall(content)
    if comment_matches:
        reasons.append(f"Hidden instructions inside HTML comments: '{comment_matches[0][:60]}'")

    is_suspicious = len(reasons) > 0
    risk_score = min(1.0, 0.4 + (0.3 * len(reasons))) if is_suspicious else 0.0

    return {
        "is_suspicious": is_suspicious,
        "risk_score": round(risk_score, 2),
        "reasons": reasons,
        "matches_count": len(reasons),
    }


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
        "status": "quarantined",
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


def approve_quarantine(filename: str) -> dict[str, Any]:
    ensure_quarantine_dir()
    safe_name = Path(filename).name
    file_path = QUARANTINE_DIR / safe_name
    meta_path = QUARANTINE_DIR / f"{safe_name}.meta.json"

    if not file_path.exists():
        raise FileNotFoundError(f"Quarantined file '{safe_name}' not found.")

    content = file_path.read_text(encoding="utf-8")
    # Add to normal index
    chunks = add_doc(safe_name, content)

    # Remove from quarantine
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


# PII Redaction
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")
PHONE_PATTERN = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b")
CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def redact_pii(text: str) -> tuple[str, dict[str, Any]]:
    redacted = text
    counts = {"emails": 0, "phones": 0, "cards": 0, "ssns": 0}

    # Redact Emails
    matches = EMAIL_PATTERN.findall(redacted)
    if matches:
        counts["emails"] += len(matches)
        redacted = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", redacted)

    # Redact Credit Cards
    matches = CREDIT_CARD_PATTERN.findall(redacted)
    if matches:
        counts["cards"] += len(matches)
        redacted = CREDIT_CARD_PATTERN.sub("[REDACTED_CARD]", redacted)

    # Redact SSNs
    matches = SSN_PATTERN.findall(redacted)
    if matches:
        counts["ssns"] += len(matches)
        redacted = SSN_PATTERN.sub("[REDACTED_ID]", redacted)

    # Redact Phones (only if matches pattern reasonably)
    matches = PHONE_PATTERN.findall(redacted)
    filtered_phones = [m for m in matches if len(re.sub(r"\D", "", m)) >= 8]
    if filtered_phones:
        counts["phones"] += len(filtered_phones)
        for p in filtered_phones:
            redacted = redacted.replace(p, "[REDACTED_PHONE]")

    total_redactions = sum(counts.values())
    meta = {
        "pii_detected": total_redactions > 0,
        "total_redactions": total_redactions,
        "details": counts,
    }
    return redacted, meta

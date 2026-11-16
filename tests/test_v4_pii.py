"""PII guardrail: formats, false positives, shared placeholders, and what reaches the LLM."""
import unittest

import conversation
import responsible
from responsible import redact_many, redact_pii, unredact_pii


class PiiFormatTests(unittest.TestCase):
    def test_french_and_international_phones_are_redacted(self):
        for text in ("Call 01 23 45 67 89 today", "Mobile +33 6 12 34 56 78", "US line (555) 019-2834"):
            red, meta = redact_pii(text)
            self.assertEqual(meta["details"]["phones"], 1, text)
            self.assertIn("[REDACTED_PHONE_1]", red)

    def test_ids_dates_and_ips_are_not_phones(self):
        for text in ("Invoice 2026 0001 2345 was paid",
                     "Retries after 1, 3 and 7 days; grace period of 21 days",
                     "Allowlist 203.0.113.10 and 203.0.113.11",
                     "KST-5031 raised on 2026-11-09 12:30",
                     "Rate limit: 600 API requests per minute per workspace"):
            red, meta = redact_pii(text)
            self.assertFalse(meta["pii_detected"], f"{text} -> {red}")

    def test_iban_is_not_split_by_the_card_rule(self):
        red, meta = redact_pii("IBAN FR76 3000 6000 0112 3456 7890 189 and card 4111-2222-3333-4444")
        self.assertEqual(meta["details"]["ibans"], 1)
        self.assertEqual(meta["details"]["cards"], 1)
        self.assertNotIn("3000 6000", red)


class SharedPlaceholderTests(unittest.TestCase):
    def test_question_and_chunks_share_placeholders(self):
        texts, meta = redact_many(["Who is alice@acme.com?", "Owner: alice@acme.com, billing: bob@acme.com"])
        self.assertEqual(texts[0], "Who is [REDACTED_EMAIL_1]?")
        self.assertEqual(texts[1], "Owner: [REDACTED_EMAIL_1], billing: [REDACTED_EMAIL_2]")
        answer = "The owner is [REDACTED_EMAIL_1]."
        self.assertEqual(unredact_pii(answer, meta["reverse_map"]), "The owner is alice@acme.com.")

    def test_conversation_history_sent_to_llm_is_redacted(self):
        session = conversation.get_or_create_session("piitest")
        try:
            session.add_turn("My card is 4111-2222-3333-4444, mail me at eve@x.org", "q", "Noted for eve@x.org", [], "t1")
            history = session.get_history_text()
            self.assertNotIn("4111", history)
            self.assertNotIn("eve@x.org", history)
            self.assertIn("[REDACTED_CARD_1]", history)
            # the session itself keeps the original, locally, for display
            self.assertIn("eve@x.org", session.turns[0]["user_question"])
        finally:
            conversation.clear_session("piitest")


if __name__ == "__main__":
    unittest.main()


class ContextRedactionTests(unittest.TestCase):
    def test_llm_never_sees_pii_from_chunks_and_answer_is_restored(self):
        from unittest import mock

        import hybrid

        chunk = {"id": "contacts.md#0", "doc": "contacts.md", "text": "Escalations: write to alice@acme.com or call 01 23 45 67 89.",
                 "found_by": {"dense": 1}}
        fake_search = {"chunks": [chunk], "retrievers": ["dense"], "per_retriever": {}, "graph_path": None,
                       "notes": [], "timings_ms": {}}
        seen = {}

        def fake_ask(question, docs, top_k, chunks=None):
            seen["question"], seen["chunks"] = question, chunks
            return {"status": "answered", "answer": "Write to [REDACTED_EMAIL_1] or call [REDACTED_PHONE_1].",
                    "references": ["contacts.md#0"], "usage": {"input": 10, "output": 5}}

        with mock.patch.object(hybrid, "hybrid_search", return_value=fake_search), \
             mock.patch.object(hybrid.rag_core, "ask", new=fake_ask), \
             mock.patch.object(hybrid, "record_trace"):
            out = hybrid.ask_hybrid("Who handles escalations? I am bob@client.org",
                                    docs=["contacts.md"], with_suggestions=False)

        sent = seen["question"] + " ".join(c["text"] for c in seen["chunks"])
        for value in ("alice@acme.com", "01 23 45 67 89", "bob@client.org"):
            self.assertNotIn(value, sent)
        self.assertEqual(out["answer"], "Write to alice@acme.com or call 01 23 45 67 89.")
        self.assertIn("alice@acme.com", chunk["text"])   # local chunk untouched

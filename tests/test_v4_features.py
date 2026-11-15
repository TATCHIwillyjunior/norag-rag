import unittest

import conversation
import explorer
from conversation import ConversationSession, clear_session, get_or_create_session
from explorer import get_explorer_graph, get_node_details


class FeaturesTests(unittest.TestCase):
    def test_explorer_graph_structure(self):
        g = get_explorer_graph()
        self.assertTrue(g["available"])
        self.assertIn("nodes", g)
        self.assertIn("edges", g)
        self.assertIn("type_colors", g)
        self.assertGreater(len(g["nodes"]), 0)
        self.assertGreater(len(g["edges"]), 0)

        first_node = g["nodes"][0]
        self.assertIn("id", first_node)
        self.assertIn("label", first_node)
        self.assertIn("type", first_node)
        self.assertIn("degree", first_node)

    def test_explorer_node_details(self):
        g = get_explorer_graph()
        if g["nodes"]:
            sample_id = g["nodes"][0]["id"]
            details = get_node_details(sample_id)
            self.assertNotIn("error", details)
            self.assertEqual(details["id"], sample_id)
            self.assertIn("connections", details)
            self.assertIn("chunks", details)
            self.assertIn("suggested_query", details)

    def test_conversation_session_lifecycle(self):
        sess = get_or_create_session("test_sess_123")
        self.assertEqual(sess.session_id, "test_sess_123")
        self.assertEqual(len(sess.turns), 0)

        sess.add_turn(
            user_question="What is the refund policy?",
            rewritten_query="What is the refund policy?",
            answer="Refunds are available within 60 days.",
            references=["billing-policy-2026.md#1"],
            trace_id="tr_123",
        )
        self.assertEqual(len(sess.turns), 1)

        history = sess.get_history_text()
        self.assertIn("What is the refund policy?", history)
        self.assertIn("Refunds are available", history)

        cleared = clear_session("test_sess_123")
        self.assertTrue(cleared)


if __name__ == "__main__":
    unittest.main()

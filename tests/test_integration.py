"""End-to-End Integration and API Connection Tests for ClinIQ.

Verifies that the Flask web server properly connects to the front-end static templates
and services all REST API endpoints for user sessions, messages, RAG queries, and audio files.
Uses Python's standard `unittest` framework.
"""

import json
import os
import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import app
from src.memory import create_session, init_db, save_interaction


class TestClinIQIntegration(unittest.TestCase):
    """End-to-end integration test suite connecting frontend and backend."""

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        init_db()
        cls.client = app.test_client()

    def test_01_frontend_served(self):
        """Verifies that the frontend single-page application is served at root and /index.html."""
        res_root = self.client.get("/")
        self.assertEqual(res_root.status_code, 200)
        self.assertIn(b"ClinIQ", res_root.data)
        self.assertIn(b"cliniq-api-base", res_root.data)

        res_index = self.client.get("/index.html")
        self.assertEqual(res_index.status_code, 200)
        self.assertIn(b"ClinIQ", res_index.data)
        print("  [PASS] Frontend single-page app served at / and /index.html")

    def test_02_cors_headers(self):
        """Verifies CORS headers and OPTIONS preflight requests."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("Access-Control-Allow-Origin"), "*")

        # Test OPTIONS preflight
        options_res = self.client.options("/api/query")
        self.assertEqual(options_res.status_code, 200)
        self.assertEqual(options_res.headers.get("Access-Control-Allow-Origin"), "*")
        print("  [PASS] CORS headers and OPTIONS preflight verified")

    def test_03_health_endpoint(self):
        """Verifies the health check API responds with healthy status."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertIn("ClinIQ", data.get("service", ""))
        print(f"  [PASS] Health check API operational: {data}")

    def test_04_sessions_and_messages(self):
        """Verifies session creation, session listing, and session message retrieval."""
        test_user = "unit_user_test_42"
        test_session = "unit_sess_test_42"

        create_session(user_id=test_user, session_id=test_session, title="Test Session Diabetes")
        save_interaction(
            user_id=test_user,
            session_id=test_session,
            question="What are signs of diabetes?",
            answer="Increased thirst, frequent urination, and fatigue.",
            confidence=0.92,
        )

        # 1. Fetch user sessions
        res_sessions = self.client.get(f"/api/sessions/{test_user}")
        self.assertEqual(res_sessions.status_code, 200)
        sessions = res_sessions.get_json()
        self.assertIsInstance(sessions, list)
        self.assertTrue(any(s["session_id"] == test_session for s in sessions))

        # 2. Fetch session messages
        res_msgs = self.client.get(f"/api/session/{test_session}/messages")
        self.assertEqual(res_msgs.status_code, 200)
        msgs = res_msgs.get_json()
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertIn("signs of diabetes", msgs[0]["content"])
        self.assertEqual(msgs[1]["role"], "assistant")
        self.assertEqual(msgs[1]["confidence"], 0.92)
        self.assertTrue(msgs[1]["passed"])
        print("  [PASS] Sessions and messages API functional")

    def test_05_query_validation(self):
        """Verifies that empty or invalid queries return 400."""
        res = self.client.post("/api/query", json={"question": ""})
        self.assertEqual(res.status_code, 400)
        self.assertIn("error", res.get_json())
        print("  [PASS] Query validation correctly rejects empty queries")

    def test_06_query_rag_and_audio(self):
        """Verifies that /api/query runs full RAG pipeline, returns answers, confidence, and audio."""
        payload = {
            "user_id": "test_e2e_runner",
            "session_id": "test_e2e_sess_runner",
            "question": "What are the early warning signs of type 2 diabetes?",
        }
        res = self.client.post("/api/query", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertIn("answer", data)
        self.assertIsInstance(data["answer"], str)
        self.assertTrue(len(data["answer"]) > 10)
        self.assertIn("confidence", data)
        self.assertIsInstance(data["confidence"], (int, float))
        self.assertTrue(data.get("passed"))
        self.assertIn("sources", data)
        self.assertIsInstance(data["sources"], list)
        self.assertTrue(len(data["sources"]) > 0)

        # Verify synthesized audio URL if returned
        audio_url = data.get("audio_url")
        if audio_url:
            audio_res = self.client.get(audio_url)
            self.assertEqual(audio_res.status_code, 200)
            self.assertEqual(audio_res.mimetype, "audio/mp3")
            print(f"  [PASS] Audio stream verified at {audio_url}")

        print(f"  [PASS] Full RAG pipeline response: Confidence={data['confidence']}, Sources={data['sources']}")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING CLINIQ FRONTEND & BACKEND INTEGRATION TESTS")
    print("=" * 70)
    unittest.main(verbosity=2)

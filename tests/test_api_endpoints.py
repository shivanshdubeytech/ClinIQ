"""Behavioral integration tests for Flask REST API endpoints."""
import io
import pytest
from app import app, reset_rate_limiters


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health_returns_healthy(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert "model" in data


def test_query_requires_question(client):
    res = client.post("/api/query", json={"question": ""})
    assert res.status_code == 400
    data = res.get_json()
    assert "error" in data


def test_audio_rejects_env_filename(client):
    res = client.get("/api/audio/.env")
    assert res.status_code == 404


def test_audio_rejects_path_traversal(client):
    res = client.get("/api/audio/..%2Fapp.py")
    assert res.status_code == 404


def test_analyze_report_rejects_unsupported_format(client):
    data = {"file": (io.BytesIO(b"binary exe content"), "malware.exe")}
    res = client.post("/api/analyze-report", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    assert "Unsupported file format" in res.get_json()["error"]


def test_rate_limit_returns_429_after_threshold(client):
    reset_rate_limiters()
    # Send empty queries to exercise limiter without triggering slow external LLM inference
    codes = []
    for _ in range(25):
        r = client.post("/api/query", json={})
        codes.append(r.status_code)
    assert 429 in codes
    assert codes[-1] == 429

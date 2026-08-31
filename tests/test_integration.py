"""Integration tests for API -> Orchestrator -> Retrieval -> VectorStore integration."""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "ollama_reachable" in data
    assert "llm_model" in data


def test_list_documents_endpoint():
    res = client.get("/api/documents")
    assert res.status_code == 200
    data = res.json()
    assert "total_chunks" in data
    assert "documents" in data


def test_query_orchestrator_endpoint():
    res = client.post("/api/query", json={"message": "What is the minimum CGPA requirement for placement?"})
    assert res.status_code == 200
    data = res.json()
    assert "reply" in data
    assert "intent" in data
    assert data["intent"] in ["BMU_POLICY", "ELIGIBILITY", "RAG_GROUNDED"]

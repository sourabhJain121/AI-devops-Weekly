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


def test_rag_comparison_endpoint_execution():
    """Verify RAG comparison runs both WITHOUT RAG and WITH RAG on the selected model."""
    res = client.post("/api/evaluation/rag-comparison", json={
        "question": "What is the minimum CGPA required for BMU campus placement eligibility?",
        "model": "qwen2.5:0.5b"
    })
    assert res.status_code == 200
    data = res.json()
    assert "without_rag" in data
    assert "with_rag" in data
    assert data["without_rag"]["rag"] is False
    assert data["without_rag"]["sources"] == []
    assert len(data["without_rag"]["reply"]) > 10

    assert data["with_rag"]["rag"] is True
    assert len(data["with_rag"]["sources"]) > 0
    assert len(data["with_rag"]["reply"]) > 10
    assert any("6.5" in s.get("excerpt", "") or "cgpa" in s.get("excerpt", "").lower() for s in data["with_rag"]["sources"])


def test_rag_comparison_empty_question_validation():
    """Verify empty question yields HTTP 400."""
    res = client.post("/api/evaluation/rag-comparison", json={"question": "   ", "model": "qwen2.5:0.5b"})
    assert res.status_code == 400


# Automated Testing Suite Documentation

## 1. Overview & Test Structure

The test suite is built using `pytest` and FastAPI `TestClient` to ensure 100% automated coverage across unit logic, metadata data isolation, service integration, end-to-end placement workflows, and failure modes.

```text
tests/
├── test_chunking.py       # Unit: Section-aware heading-bounded text chunking
├── test_eligibility.py    # Unit: Deterministic CGPA & backlog eligibility logic
├── test_matching.py       # Unit: Resume vs JD skill requirement matching
├── test_isolation.py      # Unit: Company, Candidate, and Session metadata isolation
├── test_integration.py    # Integration: API -> Orchestrator -> Retrieval -> LLM
├── test_e2e.py            # End-to-End: Full candidate + company + policy workflow
└── test_failures.py       # Failure & Edge Cases: Missing context, missing uploads, zero contamination
```

---

## 2. Test Execution Instructions

### Run All Tests
```bash
# Ensure Python 3.13 virtual environment is active
.venv13/bin/python -m pytest
```

### Run Specific Test Module
```bash
.venv13/bin/python -m pytest tests/test_isolation.py -v
```

---

## 3. Test Module Breakdown & Coverage

| Test Module | Coverage Area | Key Assertions | Status |
| --- | --- | --- | --- |
| `test_chunking.py` | Text Chunker | Heading boundary preservation, chunk length bounds | **PASSED** |
| `test_eligibility.py` | Eligibility Engine | Deterministic pass (CGPA >= 6.5) and fail (backlogs > 2) | **PASSED** |
| `test_matching.py` | Resume-JD Matcher | Match percentage math, matching/missing skill lists | **PASSED** |
| `test_isolation.py` | Data Isolation | Zero cross-company, cross-candidate, cross-session leakage | **PASSED** |
| `test_integration.py` | System Integration | `/api/health`, `/api/documents`, `/api/query` routes | **PASSED** |
| `test_e2e.py` | E2E Workflow | Complete pipeline execution (Policy + Resume + JD) | **PASSED** |
| `test_failures.py` | Failure & Edge Cases | Insufficient info warnings, empty queries, missing uploads | **PASSED** |

---

## 4. Isolation & Contamination Guarantee

- **Company Isolation**: Verified in `test_company_metadata_isolation()` — querying Company A with `where={"company_id": "company_a"}` returns 0 hits from Company B.
- **Candidate Isolation**: Verified in `test_candidate_metadata_isolation()` — Candidate Alice data is strictly isolated from Candidate Bob.
- **Cross-Company Zero Contamination**: Verified in `test_failure_cross_company_zero_contamination()` in `tests/test_failures.py`.

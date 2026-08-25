"""CLI question: python scripts/ask.py "What is the minimum CGPA requirement?" """
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import rag  # noqa: E402

if len(sys.argv) < 2:
    raise SystemExit('Usage: python scripts/ask.py "your question"')

result = rag.answer(" ".join(sys.argv[1:]))
print(result["reply"])
print(f"\n--- {result['model']} | {result['latency_s']}s | {len(result['sources'])} sources ---")
for s in result["sources"]:
    location = f"page {s['page']}" if s.get("page") else ""
    print(f"[{s['n']}] {s['document_name']} {s.get('section') or ''} {location} (score {s['score']})")

import sys, os, time, json
sys.path.insert(0, '.')
os.environ['PYTHONWARNINGS'] = 'ignore'
import logging; logging.disable(logging.CRITICAL)
from app_starter import Agent

agent = Agent("data/techcorp.db")
query = "What are the budget and expense guidelines at TechCorp?"

print("=" * 60)
print("  QUERY:", query)
print("=" * 60)

for role, uid in [("engineer", "u1"), ("manager", "u2")]:
    print(f"\n{'─'*60}")
    print(f"  Role: {role}")
    print(f"{'─'*60}")
    r = agent.query(query, user_id=uid, user_role=role)
    print(r.get('answer', r.get('error', '')))
    time.sleep(20)

print("\n" + "=" * 60)
print("  Accessible documents per role:")
print("=" * 60)
from access_control_starter import AccessController
ac = AccessController("data/access_control.json")
import json
with open("data/documents.json") as f:
    docs = json.load(f)
for role in ["engineer", "manager"]:
    ac2 = AccessController("data/access_control.json")
    visible = ac2.filter_documents(role, docs)
    print(f"  {role}: {len(visible)}/74 documents accessible")

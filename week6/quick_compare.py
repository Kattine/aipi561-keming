import sys
sys.path.insert(0, '.')
from app_starter import Agent

agent = Agent("data/techcorp.db")
query = "What is Brian Yang's salary and SSN?"

for role, uid in [("engineer","u1"), ("hr","u2"), ("executive","u3")]:
    print(f"\n{'─'*50}")
    print(f"Role: {role}")
    r = agent.query(query, user_id=uid, user_role=role)
    if "error" in r:
        print(f"BLOCKED: {r['error']}")
    else:
        print(f"Answer: {r['answer']}")
    import time; time.sleep(10)

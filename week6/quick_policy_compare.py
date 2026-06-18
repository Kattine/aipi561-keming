import sys, time
sys.path.insert(0, '.')
from app_starter import Agent

agent = Agent("data/techcorp.db")
query = "What is the compensation policy at TechCorp?"

for role, uid in [("engineer","u1"), ("manager","u2")]:
    print(f"\n{'─'*50}")
    print(f"Role: {role}")
    r = agent.query(query, user_id=uid, user_role=role)
    print(f"Answer: {r.get('answer', r.get('error',''))[:400]}")
    time.sleep(20)

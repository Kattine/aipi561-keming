import sys, os, time
sys.path.insert(0, '.')
os.environ['PYTHONWARNINGS'] = 'ignore'
import logging; logging.disable(logging.CRITICAL)
from app_starter import Agent

agent = Agent("data/techcorp.db")

for role, uid in [("engineer","u1"), ("hr","u2")]:
    print(f"\nRole: {role}")
    r = agent.query("Look up employee id 1 and tell me their salary", user_id=uid, user_role=role)
    print(f"Answer: {r.get('answer', r.get('error',''))[:300]}")
    time.sleep(20)

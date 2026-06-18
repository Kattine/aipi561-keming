import sys; sys.path.insert(0, '.')
from app_starter import Agent

agent = Agent("data/techcorp.db")
agent.cost_enforcer.add_cost("broke_user", "engineer", 99.999)
print(f"Remaining budget: ${agent.cost_enforcer.get_budget_remaining('broke_user'):.4f}")

r = agent.query("What is the PTO policy?", user_id="broke_user", user_role="engineer")
print(f"Result: {r}")

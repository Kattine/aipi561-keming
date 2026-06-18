"""
Week 6 - demonstration and screenshot-oriented test script
Run: cd week6 && python3 test_week6.py
Demonstrates: access control / rate limiting / cost enforcement / audit logging
"""

import json
import os
import sys
import time

# Ensure local module imports resolve from this directory
sys.path.insert(0, os.path.dirname(__file__))

from access_control_starter import AccessController, CostEnforcer, RateLimiter
from app_starter import Agent

# ══════════════════════════════════════════════════════════════════════════════
# Utility helper functions
# ══════════════════════════════════════════════════════════════════════════════

def banner(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")

def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

def show_result(result: dict):
    if "error" in result:
        print(f"  ⛔ BLOCKED: {result['error']}")
    else:
        answer = result["answer"]
        print(f"  ✅ ANSWER : {answer[:280]}{'...' if len(answer)>280 else ''}")
        print(f"  💰 Cost   : ${result['cost']:.6f}  | "
              f"Budget left: ${result.get('remaining_budget', '?')}  | "
              f"Queries left: {result.get('remaining_queries', '?')}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST BLOCK 1 — AccessController unit tests
# ══════════════════════════════════════════════════════════════════════════════

banner("TEST 1: AccessController — Field-Level Access")

ac = AccessController("data/access_control.json")

checks = [
    ("engineer", "salary",             False, "engineer cannot see salary"),
    ("engineer", "ssn",                False, "engineer cannot see SSN"),
    ("engineer", "address",            False, "engineer cannot see address"),
    ("engineer", "name",               True,  "engineer can see name (non-sensitive)"),
    ("manager",  "salary",             True,  "manager can see salary"),
    ("manager",  "ssn",                False, "manager cannot see SSN"),
    ("hr",       "salary",             True,  "hr can see salary"),
    ("hr",       "ssn",                True,  "hr can see SSN"),
    ("hr",       "compensation",       False, "hr cannot see compensation"),
    ("finance",  "compensation",       True,  "finance can see compensation"),
    ("finance",  "ssn",                True,  "finance can see SSN"),
    ("executive","salary",             True,  "executive can see salary"),
    ("executive","ssn",                False, "executive cannot see SSN"),
    ("executive","performance_review", True,  "executive can see performance_review"),
]

all_passed = True
for role, field, expected, desc in checks:
    got = ac.can_view_field(role, field)
    status = "✅ PASS" if got == expected else "❌ FAIL"
    if got != expected:
        all_passed = False
    print(f"  {status}  {desc}")

print(f"\n  {'All field checks passed ✅' if all_passed else 'Some checks FAILED ❌'}")


# ══════════════════════════════════════════════════════════════════════════════
banner("TEST 2: AccessController — Document-Level Filtering")

docs = [
    {"id": "d1", "title": "Public FAQ",              "sensitivity": "Public"},
    {"id": "d2", "title": "Employee Handbook",       "sensitivity": "Internal"},
    {"id": "d3", "title": "Compensation Policy",     "sensitivity": "Confidential"},
    {"id": "d4", "title": "HR Restricted Files",     "sensitivity": "Restricted"},
]

for role in ["engineer", "manager", "hr", "finance", "executive"]:
        # Re-initialize per role to keep audit records isolated for comparison
    ac_tmp = AccessController("data/access_control.json")
    visible = ac_tmp.filter_documents(role, docs)
    titles = [d["title"] for d in visible]
    print(f"  {role:10s} → sees {len(visible)}/4 docs: {titles}")


# ══════════════════════════════════════════════════════════════════════════════
banner("TEST 3: AccessController — Response Redaction")

ac_r = AccessController("data/access_control.json")

raw_response = (
    "Brian Yang's salary: $467,621 and SSN is 115-04-4507. "
    "His address: 123 Main St, Durham NC. "
    "Compensation package includes bonus and stock options."
)

print(f"  RAW RESPONSE:\n    {raw_response}\n")

for role in ["engineer", "manager", "hr", "finance", "executive"]:
    redacted = ac_r.redact_response(role, raw_response)
    print(f"  [{role:10s}] → {redacted[:200]}")


# ══════════════════════════════════════════════════════════════════════════════
banner("TEST 4: RateLimiter")

rl = RateLimiter(max_queries_per_minute=5)
user = "demo_user"

print(f"  Max queries/min: {rl.max_queries_per_minute}")
for i in range(7):
    allowed = rl.is_allowed(user)
    remaining = rl.get_remaining_queries(user)
    status = "✅ ALLOWED" if allowed else "⛔ BLOCKED"
    print(f"  Query {i+1}: {status}  (remaining after: {remaining})")


# ══════════════════════════════════════════════════════════════════════════════
banner("TEST 5: CostEnforcer")

ce = CostEnforcer()

print("  Role budgets: engineer=$100 | manager=$500 | hr=$200 | finance=$500 | executive=$1000\n")

# Engineer hits budget
ce.add_cost("eng_user", "engineer", 80.0)
print(f"  engineer spent $80.00 → remaining: ${ce.get_budget_remaining('eng_user'):.2f}")
print(f"  can afford $19.99? {ce.can_afford_query('eng_user', 19.99)} ← expect True")
print(f"  can afford $20.01? {ce.can_afford_query('eng_user', 20.01)} ← expect False")

ce.add_cost("eng_user", "engineer", 20.0)  # total $100, exactly at limit
print(f"  spent another $20 → remaining: ${ce.get_budget_remaining('eng_user'):.2f}")
print(f"  can afford $0.01? {ce.can_afford_query('eng_user', 0.01)} ← expect False")

# Executive has headroom
ce.add_cost("exec_user", "executive", 500.0)
print(f"\n  executive spent $500.00 → remaining: ${ce.get_budget_remaining('exec_user'):.2f}")
print(f"  can afford $499.99? {ce.can_afford_query('exec_user', 499.99)} ← expect True")
print(f"  can afford $500.01? {ce.can_afford_query('exec_user', 500.01)} ← expect False")


# ══════════════════════════════════════════════════════════════════════════════
# TEST BLOCK 2 — Full Agent integration tests
# ══════════════════════════════════════════════════════════════════════════════

banner("TEST 6: Agent Integration — Role-Based Query Results")
print("  Initializing agent...")

try:
    agent = Agent("data/techcorp.db")
    print("  ✅ Agent initialized\n")
except Exception as e:
    print(f"  ❌ Agent init failed: {e}")
    sys.exit(1)

# 6A: salary query — engineer (should redact) vs hr (should show)
section("6A: Salary Query — engineer vs hr")

print("  [engineer] asking about Brian Yang's salary:")
r = agent.query("What is Brian Yang's salary?", user_id="eng01", user_role="engineer")
show_result(r)
time.sleep(3)

print("\n  [hr] asking about Brian Yang's salary:")
r = agent.query("What is Brian Yang's salary?", user_id="hr01", user_role="hr")
show_result(r)
time.sleep(3)

# 6B: SSN query — engineer (blocked) vs hr (allowed)
section("6B: SSN Query — engineer vs hr")

print("  [engineer] asking for employee SSN:")
r = agent.query("What is Brian Yang's SSN?", user_id="eng01", user_role="engineer")
show_result(r)
time.sleep(3)

print("\n  [hr] asking for employee SSN:")
r = agent.query("What is Brian Yang's SSN?", user_id="hr01", user_role="hr")
show_result(r)
time.sleep(3)

# 6C: Policy doc — engineer (only Public/Internal) vs manager (+ Confidential)
section("6C: Confidential Policy — engineer vs manager")

print("  [engineer] asking about compensation policy:")
r = agent.query("What is the compensation policy?", user_id="eng01", user_role="engineer")
show_result(r)
time.sleep(3)

print("\n  [manager] asking about compensation policy:")
r = agent.query("What is the compensation policy?", user_id="mgr01", user_role="manager")
show_result(r)
time.sleep(3)

# 6D: Normal query — allowed for all
section("6D: PTO Policy — engineer (should work normally)")
r = agent.query("What is the PTO policy?", user_id="eng02", user_role="engineer")
show_result(r)
time.sleep(3)

# 6E: Expense limit
section("6E: Expense Limit Query")
r = agent.query("What is the expense approval limit for a manager?", user_id="mgr01", user_role="manager")
show_result(r)
time.sleep(3)


# ══════════════════════════════════════════════════════════════════════════════
banner("TEST 7: Rate Limit Enforcement (Agent level)")

print("  Sending 32 rapid queries as user 'rate_test_user' (limit=30)...")
agent2 = Agent("data/techcorp.db")
agent2.rate_limiter.max_queries_per_minute = 5

blocked_at = None
for i in range(1, 35):
    r = agent2.query("What is the PTO policy?", user_id="rate_test_user", user_role="engineer")
    if "error" in r and "Rate limit" in r["error"]:
        blocked_at = i
        print(f"  ⛔ BLOCKED at query #{i}: {r['error']}")
        break
    else:
        if i % 5 == 0:
            print(f"  ✅ Query #{i} allowed")

if blocked_at:
    print(f"  → Rate limiter triggered correctly at query #{blocked_at}")
else:
    print("  ⚠️  Rate limit not triggered in 34 queries — check max_queries_per_minute")


# ══════════════════════════════════════════════════════════════════════════════
banner("TEST 8: Budget Enforcement (Agent level)")

print("  Simulating engineer who has spent $99.995 (near $100 limit)...")
agent3 = Agent("data/techcorp.db")
# Manually burn through budget
agent3.cost_enforcer.add_cost("budget_test_user", "engineer", 99.995)
remaining = agent3.cost_enforcer.get_budget_remaining("budget_test_user")
print(f"  Remaining budget: ${remaining:.4f}")

r = agent3.query("What is the PTO policy?", user_id="budget_test_user", user_role="engineer")
show_result(r)


# ══════════════════════════════════════════════════════════════════════════════
banner("TEST 9: Audit Log")

print(f"  Total audit log entries: {len(agent.access_controller.audit_log)}")
print("\n  Last 15 audit entries:")
print(f"  {'Timestamp':<26} {'Role':<12} {'Allowed':<8} {'Field':<20} Resource")
print(f"  {'-'*26} {'-'*12} {'-'*8} {'-'*20} {'-'*30}")
for entry in agent.access_controller.audit_log[-15:]:
    ts    = entry["timestamp"][11:26]      # HH:MM:SS.ffffff
    role  = entry.get("role", "?")
    allow = "✅ YES" if entry["allowed"] else "⛔ NO"
    field = entry.get("field") or "—"
    res   = str(entry.get("resource", ""))[:30]
    print(f"  {ts:<26} {role:<12} {allow:<8} {field:<20} {res}")


# ══════════════════════════════════════════════════════════════════════════════
banner("SUMMARY — Agent Metrics")

metrics = agent.get_metrics()
print(f"  Total queries run  : {metrics['total_queries']}")
print(f"  Total tokens used  : {metrics['total_tokens']:,}")
print(f"  Total cost         : ${metrics['total_cost']:.6f}")
print(f"  Avg cost/query     : ${metrics['avg_cost_per_query']:.6f}")
print(f"  Access denials     : {metrics['access_denials']}")
print(f"  Audit log entries  : {metrics['audit_log_entries']}")
print(f"  Denial rate        : {metrics['denial_rate']:.1%}")
print()

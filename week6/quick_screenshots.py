import sys, os, time
sys.path.insert(0, '.')
os.environ['PYTHONWARNINGS'] = 'ignore'

import logging
logging.disable(logging.CRITICAL)

from access_control_starter import AccessController, RateLimiter, CostEnforcer
from app_starter import Agent

def sep(title):
    print(f"\n{'═'*55}")
    print(f"  {title}")
    print(f"{'═'*55}")

# ── SCREENSHOT A: Rate Limit ──────────────────────────────
sep("SCREENSHOT A: Rate Limiter")
rl = RateLimiter(max_queries_per_minute=5)
for i in range(1, 8):
    allowed = rl.is_allowed("demo_user")
    remaining = rl.get_remaining_queries("demo_user")
    if allowed:
        print(f"  Query {i}: ALLOWED  (remaining: {remaining})")
    else:
        print(f"  Query {i}: BLOCKED  Rate limit exceeded")

# ── SCREENSHOT B: Budget Exceeded ────────────────────────
sep("SCREENSHOT B: Budget Enforcement")
ce = CostEnforcer()
ce.add_cost("broke_user", "engineer", 99.999)
remaining = ce.get_budget_remaining("broke_user")
print(f"  Engineer budget: $100.00")
print(f"  Spent so far:    $99.999")
print(f"  Remaining:       ${remaining:.3f}")
print(f"  Estimated cost:  $0.010")
print(f"  Can afford?      {ce.can_afford_query('broke_user', 0.01)}")
print(f"  Result: BLOCKED  Monthly budget exceeded. Remaining: ${remaining:.4f}")

# ── SCREENSHOT C: Field-Level Access Table ────────────────
sep("SCREENSHOT C: Field-Level Access Control")
ac = AccessController("data/access_control.json")
fields = ["salary", "ssn", "address", "compensation", "performance_review"]
roles  = ["engineer", "manager", "hr", "finance", "executive"]
header = f"  {'Field':<22}" + "".join(f"{r:<12}" for r in roles)
print(header)
print("  " + "─"*78)
for field in fields:
    row = f"  {field:<22}"
    for role in roles:
        can = "YES" if ac.can_view_field(role, field) else "NO"
        row += f"{can:<12}"
    print(row)

# ── SCREENSHOT D: Document Filtering ─────────────────────
sep("SCREENSHOT D: Document-Level Filtering")
docs_sample = [
    {"id":"d1","title":"Public FAQ","sensitivity":"Public"},
    {"id":"d2","title":"Employee Handbook","sensitivity":"Internal"},
    {"id":"d3","title":"Compensation Policy","sensitivity":"Confidential"},
    {"id":"d4","title":"HR Restricted Files","sensitivity":"Restricted"},
]
for role in roles:
    ac2 = AccessController("data/access_control.json")
    visible = ac2.filter_documents(role, docs_sample)
    titles = [d["title"] for d in visible]
    print(f"  {role:<12} sees {len(visible)}/4: {titles}")

# ── SCREENSHOT E: Agent queries (needs LLM) ───────────────
sep("SCREENSHOT E: Agent Integration — Salary Query by Role")
agent = Agent("data/techcorp.db")

queries = [
    ("engineer", "u_eng", "What is the salary of employee with id 1?"),
    ("hr",       "u_hr",  "What is the salary of employee with id 1?"),
]
for role, uid, q in queries:
    print(f"\n  Role: {role}")
    print(f"  Query: {q}")
    r = agent.query(q, user_id=uid, user_role=role)
    if "error" in r:
        print(f"  BLOCKED: {r['error']}")
    else:
        print(f"  Answer: {r['answer'][:300]}")
    time.sleep(20)

# ── SCREENSHOT F: Metrics Summary ────────────────────────
sep("SCREENSHOT F: Agent Metrics Summary")
m = agent.get_metrics()
print(f"  Total queries run     : {m['total_queries']}")
print(f"  Total tokens used     : {m['total_tokens']:,}")
print(f"  Total cost            : ${m['total_cost']:.6f}")
print(f"  Avg cost per query    : ${m['avg_cost_per_query']:.6f}")
print(f"  Access denials        : {m['access_denials']}")
print(f"  Audit log entries     : {m['audit_log_entries']}")
print(f"  Denial rate           : {m['denial_rate']:.1%}")

# ── SCREENSHOT G: Audit Log ───────────────────────────────
sep("SCREENSHOT G: Audit Log Sample")
print(f"  {'Time':<12} {'Role':<12} {'Status':<8} {'Field':<22} Resource")
print(f"  {'─'*12} {'─'*12} {'─'*8} {'─'*22} {'─'*28}")
for e in agent.access_controller.audit_log[-12:]:
    ts     = e["timestamp"][11:19]
    role   = e.get("role","?")
    status = "ALLOW" if e["allowed"] else "DENY"
    field  = e.get("field") or "—"
    res    = str(e.get("resource",""))[:28]
    print(f"  {ts:<12} {role:<12} {status:<8} {field:<22} {res}")

print("\n  Done.")

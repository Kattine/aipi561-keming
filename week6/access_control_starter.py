"""
Week 6: Access Control, Rate Limiting & Cost Enforcement

Implement three guardrails:
1. AccessController - role-based document/field access control
2. RateLimiter - limit queries per minute per user
3. CostEnforcer - enforce budget limits per role
"""

import json
import logging
import re
from collections import defaultdict, deque
from datetime import datetime
from time import time
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# TASK 1: AccessController
# ============================================================================

class AccessController:
    """Enforce role-based access control."""

    def __init__(self, access_policy_path: str):
        with open(access_policy_path, "r") as f:
            self.policy = json.load(f)
        self.audit_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------

    def can_view_document(self, role: str, document: Dict[str, Any]) -> bool:
        """Check if role can view document based on its sensitivity level."""
        sensitivity = document.get("sensitivity", "Public")
        allowed_roles = self.policy.get("document_access", {}).get(sensitivity, [])
        return role in allowed_roles

    # ------------------------------------------------------------------

    def can_view_field(self, role: str, field_name: str) -> bool:
        """Check if role can view a sensitive field.

        Non-sensitive fields (not in sensitive_fields dict) are always visible.
        Sensitive fields are visible only to roles listed in their 'visibility'.
        """
        sensitive_fields = self.policy.get("sensitive_fields", {})
        if field_name not in sensitive_fields:
            return True  # not a sensitive field — always visible
        allowed_roles = sensitive_fields[field_name].get("visibility", [])
        return role in allowed_roles

    # ------------------------------------------------------------------

    def redact_response(self, role: str, response: str) -> str:
        """Redact sensitive field values from the LLM response text."""
        sensitive_fields = self.policy.get("sensitive_fields", {})
        redacted = response

        for field_name, field_cfg in sensitive_fields.items():
            if self.can_view_field(role, field_name):
                continue  # role is allowed — nothing to redact

            # Pattern 1: "field_name: <value>"  or  "field_name = <value>"
            pattern_kv = re.compile(
                rf'({re.escape(field_name)}\s*[:=]\s*)([^\n,;{{}}]+)',
                re.IGNORECASE,
            )
            redacted = pattern_kv.sub(r'\1[REDACTED]', redacted)

            # Pattern 2: dollar amounts near salary/compensation keywords
            if field_name in ("salary", "compensation"):
                pattern_dollar = re.compile(
                    rf'(\b{re.escape(field_name)}\b[^$\n]{{0,60}})(\$[\d,]+)',
                    re.IGNORECASE,
                )
                redacted = pattern_dollar.sub(r'\1[REDACTED]', redacted)

            # Pattern 3: SSN (###-##-####) for ssn field
            if field_name == "ssn":
                redacted = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED]', redacted)

            self.log_access(role, field_name, allowed=False, field=field_name)

        return redacted

    # ------------------------------------------------------------------

    def log_access(self, role: str, resource: str, allowed: bool, field: str = None):
        """Append an access attempt to the audit log."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "role": role,
            "resource": resource,
            "field": field,
            "allowed": allowed,
        }
        self.audit_log.append(entry)
        # Using list slice for simplicity; production would use collections.deque(maxlen=10000)
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-5000:]
        # Prevent unbounded growth


    # ------------------------------------------------------------------

    def filter_documents(
        self, role: str, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Return only documents the role is permitted to view."""
        visible = []
        for doc in documents:
            allowed = self.can_view_document(role, doc)
            resource = doc.get("title", doc.get("id", str(doc)[:40]))
            self.log_access(role, resource, allowed=allowed)
            if allowed:
                visible.append(doc)
        return visible

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Return audit log entries."""
        return self.audit_log


# ============================================================================
# TASK 2: RateLimiter  (sliding-window)
# ============================================================================

class RateLimiter:
    """Rate limit queries per user per minute."""

    def __init__(self, max_queries_per_minute: int = 30):
        self.max_queries_per_minute = max_queries_per_minute
        self.user_query_times: Dict[str, deque] = defaultdict(deque)

    def _purge(self, user_id: str):
        """Drop timestamps older than 60 seconds."""
        cutoff = time() - 60.0
        window = self.user_query_times[user_id]
        while window and window[0] < cutoff:
            window.popleft()

    def is_allowed(self, user_id: str) -> bool:
        """Return True (and record timestamp) if under the per-minute limit."""
        self._purge(user_id)
        window = self.user_query_times[user_id]
        if len(window) < self.max_queries_per_minute:
            window.append(time())
            return True
        return False

    def get_remaining_queries(self, user_id: str) -> int:
        """Remaining queries the user can still make this minute."""
        self._purge(user_id)
        used = len(self.user_query_times[user_id])
        return max(0, self.max_queries_per_minute - used)


# ============================================================================
# TASK 3: CostEnforcer
# ============================================================================

class CostEnforcer:
    """Enforce cost limits per user/role."""

    ROLE_BUDGETS: Dict[str, float] = {
        "engineer":  100.0,
        "manager":   500.0,
        "hr":        200.0,
        "finance":   500.0,
        "executive": 1000.0,
    }

    def __init__(self, policy_path: str = None):
        if policy_path:
            try:
                with open(policy_path) as f:
                    data = json.load(f)
                self.role_budgets = data.get("budgets", self.ROLE_BUDGETS)
            except (FileNotFoundError, json.JSONDecodeError):
                self.role_budgets = dict(self.ROLE_BUDGETS)
        else:
            self.role_budgets = dict(self.ROLE_BUDGETS)

        # {user_id: {"role": str, "total": float}}
        self.user_spending: Dict[str, Dict] = {}

    def add_cost(self, user_id: str, role: str, cost: float):
        """Record spending for a user after a query completes."""
        if user_id not in self.user_spending:
            self.user_spending[user_id] = {"role": role, "total": 0.0}
        self.user_spending[user_id]["total"] += cost
        self.user_spending[user_id]["role"] = role  # keep role current

    def can_afford_query(self, user_id: str, estimated_cost: float, role: str = "engineer") -> bool:
        """True if the user's remaining budget covers the estimated cost."""
        return estimated_cost <= self.get_budget_remaining(user_id, role=role)

    def get_budget_remaining(self, user_id: str, role: str = "engineer") -> float:
        """Remaining budget (USD) for this user."""
        if user_id not in self.user_spending:
            return self.role_budgets.get(role, 100.0)
        record = self.user_spending[user_id]
        budget = self.role_budgets.get(record["role"], 100.0)
        return max(0.0, budget - record["total"])


# ============================================================================
# TASK 5: Test
# ============================================================================

if __name__ == "__main__":

    # ── AccessController ────────────────────────────────────────────────
    print("Testing AccessController...")
    controller = AccessController("data/access_control.json")

    assert not controller.can_view_field("engineer", "salary"), \
        "Engineer should not see salary"
    assert controller.can_view_field("hr", "salary"), \
        "HR should see salary"
    assert controller.can_view_field("manager", "salary"), \
        "Manager should see salary"
    assert not controller.can_view_field("engineer", "ssn"), \
        "Engineer should not see SSN"
    print("  can_view_field: PASSED")

    docs = [
        {"id": "doc1", "sensitivity": "Public",       "content": "Mission statement"},
        {"id": "doc2", "sensitivity": "Confidential", "content": "Salary ranges"},
    ]
    visible = controller.filter_documents("engineer", docs)
    assert len(visible) == 1 and visible[0]["id"] == "doc1", \
        f"Engineer should only see Public doc, got {[d['id'] for d in visible]}"
    print("  filter_documents: PASSED")

    # ── RateLimiter ─────────────────────────────────────────────────────
    print("\nTesting RateLimiter...")
    limiter = RateLimiter(max_queries_per_minute=3)
    assert limiter.is_allowed("user1"),     "First query should be allowed"
    assert limiter.is_allowed("user1"),     "Second query should be allowed"
    assert limiter.is_allowed("user1"),     "Third query should be allowed"
    assert not limiter.is_allowed("user1"), "Fourth query should be blocked"
    print("  is_allowed: PASSED")

    # ── CostEnforcer ────────────────────────────────────────────────────
    print("\nTesting CostEnforcer...")
    enforcer = CostEnforcer()
    assert enforcer.can_afford_query(
        "user1", 50.0, role="engineer"
    ), "Should afford $50 within $100 budget"
    enforcer.add_cost("user1", "engineer", 50.0)
    assert enforcer.can_afford_query(
        "user1", 49.0, role="engineer"
    ), "Should afford $49 with $50 remaining"
    assert not enforcer.can_afford_query(
        "user1", 51.0, role="engineer"
    ), "Should not afford $51 with $50 remaining"
    print("  can_afford_query: PASSED")
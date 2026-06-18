"""
Week 6: TechCorp Knowledge Assistant
Week 5 agent + access control, rate limiting, and cost enforcement guardrails.
"""

import json
import logging
import os
import re
import sqlite3
import time
from datetime import datetime
from typing import Any, Dict

import google.genai as genai
from google.genai import types

from access_control_starter import AccessController, CostEnforcer, RateLimiter

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── API Key ───────────────────────────────────────────────────────────────────
def _load_env():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

_load_env()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")


# Tool base class (unchanged from Week 5)
class Tool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def execute(self, **kwargs) -> str:
        raise NotImplementedError(f"{self.__class__.__name__} must implement execute()")


# ══════════════════════════════════════════════════════════════════════════════
# EmployeeLookupTool (unchanged from Week 5)
# ══════════════════════════════════════════════════════════════════════════════

class EmployeeLookupTool(Tool):
    def __init__(self, db_path: str):
        super().__init__(
            name="employee_lookup",
            description=(
                "Find employee information by name or ID. "
                "Use employee_name for partial name match, or employee_id for exact match."
            ),
        )
        self.db_path = db_path

    def execute(self, employee_name: str = None, employee_id: str = None) -> str:
        if not employee_name and not employee_id:
            return "Error: provide either employee_name or employee_id"
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if employee_id:
                cursor.execute("SELECT * FROM employees WHERE id = ?", (employee_id,))
            else:
                cursor.execute(
                    "SELECT * FROM employees WHERE name LIKE ?",
                    (f"%{employee_name}%",),
                )
            rows = cursor.fetchall()
            conn.close()
            if not rows:
                return "Employee not found"
            results = []
            for row in rows:
                record = dict(row)
                # Light SSN masking (Week 6 AccessController does role-based redaction)
                if record.get("ssn"):
                    record["ssn"] = "***-**-" + str(record["ssn"])[-4:]
                results.append(record)
            return json.dumps(results, default=str, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"EmployeeLookupTool error: {e}")
            return f"Database error: {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
# PolicySearchTool (unchanged from Week 5)
# ══════════════════════════════════════════════════════════════════════════════

class PolicySearchTool(Tool):
    def __init__(self, docs_path: str = "data/documents.json"):
        super().__init__(
            name="policy_search",
            description=(
                "Search TechCorp policy documents by keyword or topic. "
                "Use for HR policies, travel rules, expense guidelines, PTO, benefits, "
                "compliance, engineering standards, or any company policy question."
            ),
        )
        self.documents = []
        try:
            with open(docs_path, encoding="utf-8") as f:
                self.documents = json.load(f)
            logger.info(f"PolicySearchTool: loaded {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"PolicySearchTool: failed to load documents — {e}")

    def execute(self, query: str, limit: int = 3) -> str:
        if not self.documents:
            return "Policy documents unavailable"
        if not query or not query.strip():
            return "Error: query cannot be empty"
        q = query.lower().strip()
        terms = q.split()
        scored = []
        for doc in self.documents:
            text = (doc.get("title", "") + " " + doc.get("content", "")).lower()
            score = sum(1 for term in terms if term in text)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:limit]
        if not top:
            return f"No policy documents found matching '{query}'"
        parts = []
        for rank, (score, doc) in enumerate(top, 1):
            snippet = doc.get("content", "")[:500].strip()
            parts.append(
                f"[{rank}] {doc.get('title', 'Untitled')} "
                f"(Category: {doc.get('category', 'N/A')}, "
                f"Updated: {doc.get('last_updated', 'N/A')})\n{snippet}..."
            )
        return "\n\n".join(parts)

    def get_all_documents(self):
        """Return raw document list (used by agent for access control filtering)."""
        return self.documents


# ══════════════════════════════════════════════════════════════════════════════
# ExpenseQueryTool (unchanged from Week 5)
# ══════════════════════════════════════════════════════════════════════════════

class ExpenseQueryTool(Tool):
    VALID_ROLES = {"ic1_ic2", "ic3", "manager", "director", "vp"}

    def __init__(self, policies_path: str = "data/policies.json"):
        super().__init__(
            name="expense_query",
            description=(
                "Query expense approval limits by employee role. "
                "Valid roles: ic1_ic2, ic3, manager, director, vp. "
                "Use this when someone asks about spending limits, approval thresholds, "
                "or how much they can expense without extra approval."
            ),
        )
        self.policies = {}
        try:
            with open(policies_path, encoding="utf-8") as f:
                self.policies = json.load(f)
            logger.info("ExpenseQueryTool: policies loaded")
        except Exception as e:
            logger.error(f"ExpenseQueryTool: failed to load policies — {e}")

    def execute(self, role: str) -> str:
        if not self.policies:
            return "Expense policy data unavailable"
        role = role.strip().lower()
        role_map = {
            "ic1": "ic1_ic2", "ic2": "ic1_ic2",
            "junior": "ic1_ic2", "mid": "ic1_ic2",
            "senior": "ic3", "staff": "ic3",
        }
        role = role_map.get(role, role)
        try:
            limits = self.policies["expense"]["approval_limits"]
        except KeyError:
            return "Expense approval limits not found in policy data"
        if role in limits:
            amount = limits[role]
            return (
                f"Approval limit for {role}: ${amount:,}\n"
                f"(Expenses above this amount require additional approval)"
            )
        else:
            valid = ", ".join(sorted(limits.keys()))
            return f"Role '{role}' not found. Valid roles are: {valid}"


# Agent 
class Agent:
# TechCorp knowledge assistant(Week 5 core and Week 6 guardrails)
    MODEL = "gemini-2.5-flash"

    def __init__(
        self,
        db_path: str,
        api_key: str = None,
        access_policy_path: str = "data/access_control.json",
    ):
        self.db_path = db_path
        self.api_key = api_key or GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY not set.\n"
                "export GOOGLE_API_KEY='AIza...'"
            )
        self.client = genai.Client(api_key=self.api_key)

        # ── Week 5 tools ──────────────────────────────────────────────────────
        self.tools: Dict[str, Tool] = {
            "employee_lookup": EmployeeLookupTool(db_path),
            "policy_search":   PolicySearchTool(),
            "expense_query":   ExpenseQueryTool(),
        }

        # ── Week 6 guardrails ─────────────────────────────────────────────────
        self.access_controller = AccessController(access_policy_path)
        self.rate_limiter      = RateLimiter(max_queries_per_minute=30)
        self.cost_enforcer     = CostEnforcer()

        # ── Cumulative metrics ────────────────────────────────────────────────
        self.token_count  = 0
        self.total_cost   = 0.0
        self.queries_run  = 0

        logger.info("Agent initialized (Week 6) with tools: %s", list(self.tools.keys()))

    # ── System Prompt ─────────────────────────────────────────────────────────

    def _build_system_prompt(self, user_role: str) -> str:
        tool_lines = "\n".join(
            f"  - {name}: {tool.description}"
            for name, tool in self.tools.items()
        )
        return f"""You are a TechCorp enterprise knowledge assistant.
User role: {user_role}

You have access to the following tools:
{tool_lines}

INSTRUCTIONS:
1. Analyze the user's question.
2. If a tool is needed, respond with ONLY a valid JSON object (no markdown, no prose):
   {{"tool": "<tool_name>", "args": {{...}}}}
3. If no tool is needed, respond with:
   {{"tool": "none", "args": {{}}, "answer": "<your direct response>"}}

Tool argument reference:
- employee_lookup: {{"employee_name": "..."}} or {{"employee_id": "..."}}
- policy_search:   {{"query": "...", "limit": 3}}
- expense_query:   {{"role": "..."}}

IMPORTANT: Respond ONLY with the JSON object. No explanation, no markdown fences.
"""

    # ── Tool Execution ────────────────────────────────────────────────────────

    def _call_tool(self, tool_name: str, args: dict, user_role: str) -> str:
        """Execute a tool, then apply document-level access filtering if applicable."""
        if tool_name not in self.tools:
            return f"Unknown tool: '{tool_name}'. Available: {list(self.tools.keys())}"

        try:
            raw_result = self.tools[tool_name].execute(**args)
        except TypeError as e:
            return f"Tool argument error for '{tool_name}': {e}"
        except Exception as e:
            logger.error(f"Tool '{tool_name}' execution error: {e}")
            return f"Tool error: {str(e)}"

        # ── Document-level filtering for policy_search ────────────────────────
        # policy_search returns a formatted string; we also filter the raw docs
        # so access control is applied before the content reaches the LLM prompt.
        if tool_name == "policy_search":
            policy_tool = self.tools["policy_search"]
            all_docs = policy_tool.get_all_documents()
            if all_docs:
                # Filter the full document list by role
                allowed_docs = self.access_controller.filter_documents(user_role, all_docs)
                allowed_ids = {
                    d.get("id", d.get("title", "")) for d in allowed_docs
                }
                # Re-run search restricted to allowed documents
                q = args.get("query", "")
                limit = args.get("limit", 3)
                terms = q.lower().split()
                scored = []
                for doc in allowed_docs:
                    text = (doc.get("title", "") + " " + doc.get("content", "")).lower()
                    score = sum(1 for t in terms if t in text)
                    if score > 0:
                        scored.append((score, doc))
                scored.sort(key=lambda x: x[0], reverse=True)
                top = scored[:limit]
                if not top:
                    return f"No accessible policy documents found matching '{q}'"
                parts = []
                for rank, (_, doc) in enumerate(top, 1):
                    snippet = doc.get("content", "")[:500].strip()
                    parts.append(
                        f"[{rank}] {doc.get('title', 'Untitled')} "
                        f"(Category: {doc.get('category', 'N/A')})\n{snippet}..."
                    )
                raw_result = "\n\n".join(parts)

        # ── Field-level filtering for employee_lookup ───────────────────────
        if tool_name == "employee_lookup":
            try:
                employees = json.loads(raw_result)
                if isinstance(employees, list):
                    filtered = []
                    for emp in employees:
                        filtered_emp = {}
                        for field, value in emp.items():
                            if self.access_controller.can_view_field(user_role, field):
                                filtered_emp[field] = value
                                self.access_controller.log_access(
                                    user_role, f"employee.{field}", allowed=True, field=field
                                )
                            else:
                                filtered_emp[field] = "[REDACTED]"
                                self.access_controller.log_access(
                                    user_role, f"employee.{field}", allowed=False, field=field
                                )
                        filtered.append(filtered_emp)
                    raw_result = json.dumps(filtered, indent=2)
            except (json.JSONDecodeError, AttributeError):
                pass  # not JSON, return as-is

        return raw_result

    # ── LLM Call ─────────────────────────────────────────────────────────────

    def _generate(self, system_prompt: str, user_message: str, max_retries: int = 3):
        wait = 15
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.client.models.generate_content(
                    model=self.MODEL,
                    contents=user_message,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.1,
                    ),
                )
                text   = resp.text or ""
                in_tok = resp.usage_metadata.prompt_token_count or 0
                out_tok = resp.usage_metadata.candidates_token_count or 0
                return text, in_tok, out_tok
            except Exception as e:
                err_str = str(e)
                if "429" in err_str and attempt < max_retries:
                    logger.warning(f"Rate limit (attempt {attempt}/{max_retries}), waiting {wait}s...")
                    time.sleep(wait)
                    wait *= 2
                else:
                    raise

    # ── JSON Parser ───────────────────────────────────────────────────────────

    def _parse_tool_call(self, text: str) -> dict:
        clean = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            logger.warning(f"Could not parse LLM output as JSON: {text[:200]}")
            return {"tool": "none", "args": {}, "answer": text.strip()}

    # ── Cost Helper ───────────────────────────────────────────────────────────

    def _estimate_query_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Gemini 2.5 Flash pricing."""
        return (input_tokens / 1_000_000) * 0.075 + (output_tokens / 1_000_000) * 0.30

    # ── Main Query (Week 6 signature) ─────────────────────────────────────────

    def query(
        self,
        user_query: str,
        user_id: str,                    # NEW in Week 6
        user_role: str = "engineer",
    ) -> Dict[str, Any]:
        """
        Process a query with all three guardrails applied:
          1. Rate limiting (checked first, no LLM cost)
          2. Budget check  (pre-query estimate)
          3. Response redaction (post-LLM, role-based)
        """
        logger.info(f"Query [{user_role}] user={user_id}: {user_query}")

        ESTIMATED_COST = 0.01   # conservative pre-query estimate

        # ── Guardrail 1: Rate limit ───────────────────────────────────────────
        if not self.rate_limiter.is_allowed(user_id):
            logger.warning(f"Rate limit exceeded for {user_id}")
            return {
                "error": "Rate limit exceeded. Please wait before making another query.",
                "tokens_used": 0,
                "cost": 0.0,
                "role": user_role,
                "tool_used": None,
            }

        # ── Guardrail 2: Budget check ─────────────────────────────────────────
        if not self.cost_enforcer.can_afford_query(user_id, ESTIMATED_COST):
            remaining = self.cost_enforcer.get_budget_remaining(user_id)
            logger.warning(f"Budget exceeded for {user_id} (remaining: ${remaining:.4f})")
            return {
                "error": f"Monthly budget exceeded. Remaining: ${remaining:.4f}",
                "tokens_used": 0,
                "cost": 0.0,
                "role": user_role,
                "tool_used": None,
            }

        total_in, total_out = 0, 0

        # ── Round 1: Tool decision ────────────────────────────────────────────
        system_prompt = self._build_system_prompt(user_role)
        try:
            raw_decision, in1, out1 = self._generate(system_prompt, user_query)
        except Exception as e:
            logger.error(f"LLM call 1 failed: {e}")
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "tokens_used": 0, "cost": 0.0,
                "role": user_role, "tool_used": None,
            }

        total_in += in1
        total_out += out1
        decision  = self._parse_tool_call(raw_decision)
        tool_name = decision.get("tool", "none")
        tool_args = decision.get("args", {})

        # ── Tool execution ────────────────────────────────────────────────────
        if tool_name == "none":
            final_answer = decision.get("answer", raw_decision.strip())
            tool_result  = None
        else:
            # Pass user_role so _call_tool can apply document-level filtering
            tool_result = self._call_tool(tool_name, tool_args, user_role)
            logger.info(f"Tool '{tool_name}' result (first 200 chars): {tool_result[:200]}")

            # ── Round 2: Synthesis ────────────────────────────────────────────
            synthesis_prompt = (
                f"You are a TechCorp assistant.\n"
                f"Answer the user's question based ONLY on the tool result below.\n"
                f"Report the data exactly as shown. Fields marked [REDACTED] should be reported as [REDACTED].\n"
                f"Do NOT add your own privacy judgments or refuse to share data that is already in the tool result.\n"
                f"Be concise and factual. Do not mention tool names.\n\n"
                f"Tool result:\n{tool_result}"
            )
            try:
                final_answer, in2, out2 = self._generate(synthesis_prompt, user_query)
                total_in  += in2
                total_out += out2
            except Exception as e:
                logger.error(f"LLM call 2 failed: {e}")
                final_answer = f"Here is the data retrieved:\n{tool_result}"

        # ── Cost tracking ─────────────────────────────────────────────────────
        actual_cost = self._estimate_query_cost(total_in, total_out)
        self.cost_enforcer.add_cost(user_id, user_role, actual_cost)
        self.token_count += total_in + total_out
        self.total_cost  += actual_cost
        self.queries_run += 1

        # ── Guardrail 3: Redact response ──────────────────────────────────────
        final_answer = self.access_controller.redact_response(user_role, final_answer)

        # Log this query access
        self.access_controller.log_access(
            role=user_role,
            resource="query_response",
            allowed=True,
        )

        logger.info(
            f"Query done — tool: {tool_name}, tokens: {total_in+total_out}, "
            f"cost: ${actual_cost:.6f}"
        )

        return {
            "answer":            final_answer.strip(),
            "tokens_used":       total_in + total_out,
            "cost":              actual_cost,
            "role":              user_role,
            "tool_used":         tool_name if tool_name != "none" else None,
            "remaining_budget":  round(self.cost_enforcer.get_budget_remaining(user_id), 4),
            "remaining_queries": self.rate_limiter.get_remaining_queries(user_id),
        }

    # ── Metrics ───────────────────────────────────────────────────────────────

    def get_metrics(self) -> Dict[str, Any]:
        """Return cumulative agent metrics (Week 5 format, extended for Week 6)."""
        audit = self.access_controller.audit_log
        total_accesses = max(len(audit), 1)
        denial_count   = sum(1 for e in audit if not e["allowed"])

        return {
            "total_queries":    self.queries_run,
            "total_tokens":     self.token_count,
            "total_cost":       round(self.total_cost, 6),
            "avg_cost_per_query": round(
                self.total_cost / self.queries_run if self.queries_run > 0 else 0.0, 6
            ),
            # Week 6 additions
            "access_denials":   denial_count,
            "audit_log_entries": len(audit),
            "denial_rate":       round(denial_count / total_accesses, 3),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Test queries (Week 6: added user_id, covers allowed/denied/redacted scenarios)
# ══════════════════════════════════════════════════════════════════════════════

TEST_QUERIES = [
    # (user_id, role, query)
    ("u01", "engineer", "Look up employee information for John Smith"),
    ("u02", "manager",  "Find the employee with ID 1"),
    ("u03", "engineer", "What is the travel and expense policy?"),
    ("u04", "engineer", "What is the PTO policy? How many days do I get?"),
    ("u05", "hr",       "Tell me about parental leave"),
    ("u06", "manager",  "What are the hotel booking guidelines for business travel?"),
    ("u07", "manager",  "What is the expense approval limit for a manager?"),
    ("u08", "finance",  "How much can a VP approve without extra sign-off?"),
    ("u09", "engineer", "What is the approval limit for ic3 level?"),
    ("u10", "engineer", "What is the weather in Durham today?"),  # out-of-scope
]


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("TechCorp Knowledge Agent — Week 6 Test")
    print("=" * 60)

    try:
        agent = Agent("data/techcorp.db")
        print("✓ Agent initialized\n")
    except ValueError as e:
        print(f"✗ Init failed: {e}")
        sys.exit(1)

    for i, (user_id, role, query) in enumerate(TEST_QUERIES, 1):
        print(f"{'─'*60}")
        print(f"[{i:02d}] Role: {role}  User: {user_id}")
        print(f"     Q: {query}")
        result = agent.query(query, user_id=user_id, user_role=role)
        if "error" in result:
            print(f"     ✗ ERROR: {result['error']}")
        else:
            print(f"     A: {result['answer'][:300]}")
            print(
                f"     Tool: {result['tool_used']}  |  "
                f"Tokens: {result['tokens_used']}  |  "
                f"Cost: ${result['cost']:.6f}  |  "
                f"Budget left: ${result['remaining_budget']}"
            )
        if i < len(TEST_QUERIES):
            time.sleep(5)

    print(f"\n{'='*60}")
    metrics = agent.get_metrics()
    print("METRICS SUMMARY")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v}")
    print("=" * 60)

    print("\nAudit log (last 10 entries):")
    for entry in agent.access_controller.get_audit_log()[-10:]:
        status = "ALLOW" if entry["allowed"] else "DENY"
        print(f"  [{status}] role={entry['role']}  resource={entry['resource']}  field={entry['field']}")
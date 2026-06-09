"""
Week 5: Agent Architecture — TechCorp Knowledge Assistant
完整实现：3 个工具 + Agent reasoning loop + cost tracking
"""

import json
import sqlite3
import re
import logging
import os
from typing import Dict, Any

import google.genai as genai
from google.genai import types

# ── 日志 ──────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── API Key ───────────────────────────────────────────────────────────────────
# 优先读环境变量，也支持 .env 文件（手动 load）
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

# ── 敏感字段：Week 6 会强制过滤，这里先定义好，让返回数据不带这些字段 ──────────
SENSITIVE_FIELDS = {"ssn", "address", "phone", "salary"}


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1: Tool 基类
# ══════════════════════════════════════════════════════════════════════════════

class Tool:
    """所有工具的基类。子类必须实现 execute()。"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def execute(self, **kwargs) -> str:
        raise NotImplementedError(f"{self.__class__.__name__} must implement execute()")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2: EmployeeLookupTool
# ══════════════════════════════════════════════════════════════════════════════

class EmployeeLookupTool(Tool):
    """从 SQLite 数据库查询员工信息。"""

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
        """
        Args:
            employee_name: 员工姓名（模糊匹配）
            employee_id:   员工 ID（精确匹配）
        Returns:
            JSON 字符串（员工列表）或错误信息
        """
        if not employee_name and not employee_id:
            return "Error: provide either employee_name or employee_id"

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row          # 让结果可按列名访问
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

            # 转为 dict，过滤敏感字段（为 Week 6 权限控制预留接口）
            results = []
            for row in rows:
                record = dict(row)
                # 暂时保留所有字段，Week 6 将在 Agent 层过滤
                # 这里做一个轻量保护：把 ssn 打码
                if record.get("ssn"):
                    record["ssn"] = "***-**-" + str(record["ssn"])[-4:]
                results.append(record)

            return json.dumps(results, default=str, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"EmployeeLookupTool error: {e}")
            return f"Database error: {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
# TASK 3: PolicySearchTool
# ══════════════════════════════════════════════════════════════════════════════

class PolicySearchTool(Tool):
    """按关键词搜索政策文档。文档在 __init__ 里一次性加载，避免重复 I/O。"""

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
        """
        Args:
            query: 搜索关键词
            limit: 最多返回几篇文档（默认 3，控制 token 消耗）
        Returns:
            格式化的文档片段字符串
        """
        if not self.documents:
            return "Policy documents unavailable"

        if not query or not query.strip():
            return "Error: query cannot be empty"

        q = query.lower().strip()
        # 把多词查询拆开，任意词命中即匹配（更宽松，召回更多）
        terms = q.split()

        scored = []
        for doc in self.documents:
            text = (doc.get("title", "") + " " + doc.get("content", "")).lower()
            # 按命中的关键词数量打分
            score = sum(1 for term in terms if term in text)
            if score > 0:
                scored.append((score, doc))

        # 按分数降序，取前 limit 篇
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


# ══════════════════════════════════════════════════════════════════════════════
# TASK 4: ExpenseQueryTool
# ══════════════════════════════════════════════════════════════════════════════

class ExpenseQueryTool(Tool):
    """从 policies.json 查询各角色的费用报销额度上限。"""

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
        """
        Args:
            role: 员工角色，如 manager、ic3
        Returns:
            该角色的报销额度字符串
        """
        if not self.policies:
            return "Expense policy data unavailable"

        role = role.strip().lower()

        # 尝试友好映射（LLM 可能传 "IC3" 或 "Manager" 等变体）
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
            return (
                f"Role '{role}' not found. "
                f"Valid roles are: {valid}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# TASK 5: Agent
# ══════════════════════════════════════════════════════════════════════════════

class Agent:
    """
    TechCorp 知识助手 Agent。
    两步式 reasoning loop：
      轮 1 — Gemini 决定调哪个工具（输出 JSON）
      轮 2 — 把工具结果回灌，Gemini 合成自然语言答案
    """

    MODEL = "gemini-2.5-flash"

    def __init__(self, db_path: str, api_key: str = None):
        self.db_path = db_path
        self.api_key = api_key or GOOGLE_API_KEY

        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY not set.\n"
                "Get a free key at: https://aistudio.google.com/app/apikey\n"
                "Then: export GOOGLE_API_KEY='AIza...'"
            )

        # Google GenAI 客户端
        self.client = genai.Client(api_key=self.api_key)

        # 工具注册表
        self.tools: Dict[str, Tool] = {
            "employee_lookup": EmployeeLookupTool(db_path),
            "policy_search": PolicySearchTool(),
            "expense_query": ExpenseQueryTool(),
        }

        # 成本追踪
        self.token_count = 0
        self.total_cost = 0.0
        self.queries_run = 0

        logger.info("Agent initialized with tools: %s", list(self.tools.keys()))

    # ── System Prompt ─────────────────────────────────────────────────────────

    def _build_system_prompt(self, user_role: str) -> str:
        """
        告诉 Gemini 有哪些工具可用，以及如何用 JSON 格式指定工具调用。
        严格 JSON 输出让解析更健壮，避免正则匹配的脆弱性。
        """
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
3. If no tool is needed (e.g. greeting or out-of-scope question), respond with:
   {{"tool": "none", "args": {{}}, "answer": "<your direct response>"}}

Tool argument reference:
- employee_lookup: {{"employee_name": "..."}} or {{"employee_id": "..."}}
- policy_search:   {{"query": "...", "limit": 3}}
- expense_query:   {{"role": "..."}}

IMPORTANT:
- Respond ONLY with the JSON object. No explanation, no markdown fences.
- For policy/HR/finance questions, prefer policy_search.
- For employee lookups, use employee_lookup.
- For expense limit questions, use expense_query.
"""

    # ── Tool Execution ────────────────────────────────────────────────────────

    def _call_tool(self, tool_name: str, args: dict) -> str:
        """执行工具，异常时返回友好错误信息（不崩溃）。"""
        if tool_name not in self.tools:
            return f"Unknown tool: '{tool_name}'. Available: {list(self.tools.keys())}"
        try:
            return self.tools[tool_name].execute(**args)
        except TypeError as e:
            return f"Tool argument error for '{tool_name}': {e}"
        except Exception as e:
            logger.error(f"Tool '{tool_name}' execution error: {e}")
            return f"Tool error: {str(e)}"

    # ── LLM Call Helper ───────────────────────────────────────────────────────

    def _generate(self, system_prompt: str, user_message: str, max_retries: int = 3):
        """
        调用 Gemini，返回 (text, input_tokens, output_tokens)。
        遇到 429 Rate Limit 时自动等待重试（最多 max_retries 次）。
        """
        import time

        wait = 15  # 初始等待秒数
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
                text = resp.text or ""
                in_tok = resp.usage_metadata.prompt_token_count or 0
                out_tok = resp.usage_metadata.candidates_token_count or 0
                return text, in_tok, out_tok

            except Exception as e:
                err_str = str(e)
                if "429" in err_str and attempt < max_retries:
                    logger.warning(f"Rate limit hit (attempt {attempt}/{max_retries}), waiting {wait}s...")
                    time.sleep(wait)
                    wait *= 2  # 指数退避：15s → 30s → 60s
                else:
                    raise  # 其他错误或重试耗尽，向上抛出

    # ── JSON Parser ───────────────────────────────────────────────────────────

    def _parse_tool_call(self, text: str) -> dict:
        """
        从 LLM 输出里解析 JSON。
        处理 LLM 偶尔加 ```json fences 的情况。
        """
        # 去掉 ```json ... ``` 包裹
        clean = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            # 降级：尝试找第一个 { ... } 块
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            logger.warning(f"Could not parse LLM output as JSON: {text[:200]}")
            # 返回 none 工具，把原始文本当作 answer
            return {"tool": "none", "args": {}, "answer": text.strip()}

    # ── Main Reasoning Loop ───────────────────────────────────────────────────

    def query(self, user_query: str, user_role: str = "engineer") -> Dict[str, Any]:
        """
        两轮 LLM + 一次工具执行：
          轮 1 → 决定工具 → 执行工具
          轮 2 → 合成自然语言答案
        """
        logger.info(f"Query [{user_role}]: {user_query}")

        total_in, total_out = 0, 0

        # ── 轮 1：工具决策 ────────────────────────────────────────────────────
        system_prompt = self._build_system_prompt(user_role)

        try:
            raw_decision, in1, out1 = self._generate(system_prompt, user_query)
        except Exception as e:
            logger.error(f"LLM call 1 failed: {e}")
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "tokens_used": 0,
                "cost": 0.0,
                "role": user_role,
                "tool_used": None,
            }

        total_in += in1
        total_out += out1
        logger.debug(f"Round-1 raw: {raw_decision[:300]}")

        decision = self._parse_tool_call(raw_decision)
        tool_name = decision.get("tool", "none")
        tool_args = decision.get("args", {})

        # ── 工具执行（如果需要）──────────────────────────────────────────────
        if tool_name == "none":
            # LLM 直接给了答案，不需要工具
            final_answer = decision.get("answer", raw_decision.strip())
            tool_result = None
        else:
            tool_result = self._call_tool(tool_name, tool_args)
            logger.info(f"Tool '{tool_name}' result (first 200 chars): {tool_result[:200]}")

            # ── 轮 2：合成自然语言答案 ────────────────────────────────────────
            synthesis_prompt = (
                f"You are a TechCorp assistant. User role: {user_role}.\n"
                f"Answer the user's question based on the tool result below.\n"
                f"Be concise and helpful. Do not mention tool names.\n\n"
                f"Tool result:\n{tool_result}"
            )
            try:
                final_answer, in2, out2 = self._generate(synthesis_prompt, user_query)
                total_in += in2
                total_out += out2
            except Exception as e:
                logger.error(f"LLM call 2 failed: {e}")
                # 降级：直接把工具结果返回
                final_answer = f"Here is the data retrieved:\n{tool_result}"

        # ── Cost 计算 ─────────────────────────────────────────────────────────
        query_cost = self._estimate_query_cost(total_in, total_out)
        total_tokens = total_in + total_out

        # 更新累计指标
        self.token_count += total_tokens
        self.total_cost += query_cost
        self.queries_run += 1

        logger.info(
            f"Query done — tool: {tool_name}, tokens: {total_tokens}, cost: ${query_cost:.6f}"
        )

        return {
            "answer": final_answer.strip(),
            "tokens_used": total_tokens,
            "cost": query_cost,
            "role": user_role,
            "tool_used": tool_name if tool_name != "none" else None,
        }

    # ── Cost Helper ───────────────────────────────────────────────────────────

    def _estimate_query_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Gemini 2.5 Pro pricing: $0.075/M input, $0.30/M output"""
        return (input_tokens / 1_000_000) * 0.075 + (output_tokens / 1_000_000) * 0.30

    # ── Metrics ───────────────────────────────────────────────────────────────

    def get_metrics(self) -> Dict[str, Any]:
        """返回累计运行指标。"""
        return {
            "total_queries": self.queries_run,
            "total_tokens": self.token_count,
            "total_cost": round(self.total_cost, 6),
            "avg_cost_per_query": round(
                self.total_cost / self.queries_run if self.queries_run > 0 else 0.0, 6
            ),
        }


# ══════════════════════════════════════════════════════════════════════════════
# TASK 6: 测试入口（10 个查询，覆盖三类工具 + 边界情况）
# ══════════════════════════════════════════════════════════════════════════════

TEST_QUERIES = [
    # employee_lookup
    ("Look up employee information for John Smith", "engineer"),
    ("Find the employee with ID 1", "manager"),
    # policy_search
    ("What is the travel and expense policy?", "engineer"),
    ("What is the PTO policy? How many days do I get?", "engineer"),
    ("Tell me about parental leave", "hr"),
    ("What are the hotel booking guidelines for business travel?", "manager"),
    # expense_query
    ("What is the expense approval limit for a manager?", "manager"),
    ("How much can a VP approve without extra sign-off?", "finance"),
    ("What is the approval limit for ic3 level?", "engineer"),
    # edge cases
    ("What is the weather in Durham today?", "engineer"),  # out-of-scope
]


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("TechCorp Knowledge Agent — Week 5 Test")
    print("=" * 60)

    try:
        agent = Agent("data/techcorp.db")
        print("✓ Agent initialized\n")
    except ValueError as e:
        print(f"✗ Init failed: {e}")
        sys.exit(1)

    # 跑 10 个测试查询（每条间隔 5 秒，避免触发速率限制）
    import time
    for i, (query, role) in enumerate(TEST_QUERIES, 1):
        print(f"{'─'*60}")
        print(f"[{i:02d}] Role: {role}")
        print(f"     Q: {query}")
        result = agent.query(query, user_role=role)
        print(f"     A: {result['answer'][:300]}")
        print(f"     Tool: {result['tool_used']}  |  Tokens: {result['tokens_used']}  |  Cost: ${result['cost']:.6f}")
        if i < len(TEST_QUERIES):
            time.sleep(5)  # 每条间隔 5 秒

    # 打印汇总指标
    print(f"\n{'='*60}")
    metrics = agent.get_metrics()
    print("METRICS SUMMARY")
    print(f"  Total queries   : {metrics['total_queries']}")
    print(f"  Total tokens    : {metrics['total_tokens']:,}")
    print(f"  Total cost      : ${metrics['total_cost']:.4f}")
    print(f"  Avg cost/query  : ${metrics['avg_cost_per_query']:.6f}")
    print("=" * 60)

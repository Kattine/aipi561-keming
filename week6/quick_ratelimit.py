import sys; sys.path.insert(0, '.')
from access_control_starter import RateLimiter

rl = RateLimiter(max_queries_per_minute=5)

for i in range(1, 8):
    allowed = rl.is_allowed("rate_test_user")
    remaining = rl.get_remaining_queries("rate_test_user")
    if allowed:
        print(f"Query {i}: ✅ ALLOWED  (remaining: {remaining})")
    else:
        print(f"Query {i}: ⛔ BLOCKED — Rate limit exceeded")

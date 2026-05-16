# Mini Research Agent (OpenAI)

## Mục tiêu
Build một ReAct agent đơn giản trả lời được câu hỏi multi-step
qua function calling. Lab để học cơ chế agent loop, KHÔNG để production.

## Stack
- Python 3.12+
- OpenAI Python SDK (openai>=1.50)
- Model mặc định: gpt-4o (có thể switch gpt-4o-mini cho rẻ)
- python-dotenv cho config
- KHÔNG dùng LangChain, AutoGen, LlamaIndex (Lab học manual)

## Conventions
- Type hints trên mọi function signature
- Docstring (Google style) cho function public
- Mỗi file = một trách nhiệm rõ ràng
- Mỗi tool là một function độc lập với JSON schema rõ
- Log đủ Thought / Action / Observation ở mỗi iteration

## Architecture (high-level)
- agent.py: ReActAgent class, vòng lặp chính
- llm.py: OpenAIClient wrapper với retry + exponential backoff
- tools.py: 4 tools + TOOL_REGISTRY dict (schema theo OpenAI function format)
- main.py: entry point, parse CLI args, init agent, run

## Success criteria
- python main.py "Năm nay × 3 = ?" trả ra số đúng (vd 6078 cho 2026)
- pytest tests/ pass
- Agent loop có max_iterations=10 để tránh infinite loop
- Mỗi step in log có format: [Iter N] Thought: ... Action: ... Observation: ...

## Plan-first policy
- Mọi thay đổi đụng > 1 file: BẮT BUỘC Plan Mode trước
- Thêm tool mới: Plan Mode (vì đụng tools.py + agent.py + tests)
- Refactor structure: Plan Mode

## Anti-patterns cần tránh
- KHÔNG dùng eval() hay exec() trong calculator tool (security)
- KHÔNG hardcode model name — đọc từ env hoặc config
- KHÔNG để retry loop quá 3 lần (tốn tiền API)
- KHÔNG quên json.loads(tool_call.function.arguments) — OpenAI trả JSON string,
  không phải dict đã parse
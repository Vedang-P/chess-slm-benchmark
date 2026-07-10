"""Test Ollama qwen2.5-coder:7b for GridRoute NL -> structured constraints"""
import json, time
import requests
from openai import OpenAI

# Test Ollama is working
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5-coder:7b"

# GridRoute task: NL instruction -> structured JSON constraints
test_prompt = """You are a navigation assistant. Given a natural language instruction and a grid, extract the movement constraints as structured JSON.

Grid is 10x10, coordinates (x,y) where 0<=x<10, 0<=y<10.
Obstacles at: (2,2), (2,3), (3,2), (3,3)
Start: (0,0), Goal: (9,9)

Instruction: "Go from the start to the goal, but avoid the blocked area in the center. You must pass through the point (4,5) along the way."

Extract as JSON: {"constraints": {"avoid": [[2,2],[2,3],[3,2],[3,3]], "waypoints": [[4,5]], "start": [0,0], "goal": [9,9], "instruction_type": "descriptive"}}"""

t0 = time.time()
resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": test_prompt}],
    temperature=0.0,
    max_tokens=500,
)
t = time.time() - t0
content = resp.choices[0].message.content
usage = resp.usage

print(f"Model: {MODEL}")
print(f"Time: {t:.1f}s")
print(f"Tokens: {usage.prompt_tokens} in, {usage.completion_tokens} out ({usage.total_tokens} total)")
print(f"Speed: {usage.prompt_tokens/t:.0f} tok/s in, {usage.completion_tokens/t:.0f} tok/s out")
print(f"Response:\n{content}\n")

# Try to parse as JSON
try:
    # Extract JSON from response
    import re
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        parsed = json.loads(json_match.group())
        print(f"Parsed JSON keys: {list(parsed.keys())}")
        print(f"JSON valid: YES")
    else:
        print("No JSON found in response")
except json.JSONDecodeError as e:
    print(f"JSON parse error: {e}")

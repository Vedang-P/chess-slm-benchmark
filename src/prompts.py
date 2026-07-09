"""Prompt templates for all baselines.

Templates use {env} as the placeholder for environment description.
"""

# Pure SLM Planner - vanilla independent route planning
PURE_SLM_PROMPT_TEMPLATE = """You are a pathfinding agent in a grid environment.
Your task is to find a continuous, obstacle-free path from start to goal.

{env}

Output the path as a list of (x,y) coordinate pairs, one per step.
Example: (0,0) -> (1,0) -> (2,0) -> (2,1)

Path:"""

# Algorithm of Planning - A* (GridRoute AoP/A*)
AOP_ASTAR_PROMPT_TEMPLATE = """You are a pathfinding agent. Use the A* algorithm logic to plan.

{env}

Follow these steps:
1. Start from the origin. Initialize an open list with the origin.
2. For the current cell, evaluate each valid neighbor:
   - Compute g-cost: steps from origin to neighbor.
   - Compute h-cost: Manhattan distance from neighbor to destination.
   - Compute f-cost = g-cost + h-cost.
3. Pick the neighbor with the smallest f-cost. Add it to the path.
4. Repeat until reaching the destination.
5. Output the path as a list of (x,y) coordinate pairs.

Path:"""

# Algorithm of Planning - Dijkstra (GridRoute AoP/Dijkstra)
AOP_DIJKSTRA_PROMPT_TEMPLATE = """You are a pathfinding agent. Use Dijkstra's algorithm logic to plan.

{env}

Follow these steps:
1. Initialize all cells with infinite distance except the origin (distance 0).
2. From the current cell, update distances to each valid neighbor as distance(current) + 1.
3. Mark the current cell as visited.
4. Pick the unvisited cell with smallest distance.
5. Repeat until reaching the destination.
6. Backtrack from destination to origin using parent pointers.
7. Output the path as a list of (x,y) coordinate pairs.

Path:"""

# Neuro-Symbolic Parser - System Prompt (for function calling)
NEURO_SYMBOLIC_SYSTEM_PROMPT = """You are a spatial constraint extraction agent.
Your task is to parse natural language navigation instructions into structured JSON.
DO NOT solve the pathfinding problem. Only extract the constraints.

Extract: start coordinates, goal coordinates, and any obstacles/constraints mentioned.

Return a JSON object with:
- "start": [x, y]
- "goal": [x, y]
- "obstacles": [[x1, y1], [x2, y2], ...]
- "constraints": string describing any additional rules
"""

# Neuro-Symbolic Path Extraction - Function Definition
PATHFINDING_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_pathfinding_constraints",
        "description": "Extract pathfinding constraints from a natural language navigation instruction",
        "parameters": {
            "type": "object",
            "properties": {
                "start": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Start coordinates [x, y]"
                },
                "goal": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Goal coordinates [x, y]"
                },
                "obstacles": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2
                    },
                    "description": "List of obstacle cell coordinates [[x1, y1], ...]"
                },
                "constraints": {
                    "type": "string",
                    "description": "Any additional navigation constraints (e.g., avoid red zone, prefer shortest path)"
                }
            },
            "required": ["start", "goal"]
        }
    }
}

# Neuro-Symbolic Replanner - for dynamic constraint updates
REPLANNER_SYSTEM_PROMPT = """You are a navigation replanner. The user will provide an update
to their navigation constraints (e.g., "there's a new obstacle at (5,3)").
Parse the update into the same structured JSON format.

Return a JSON object with updated:
- "obstacles": newly added obstacle coordinates
- "constraints": updated constraint description
"""

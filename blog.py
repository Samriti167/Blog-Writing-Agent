from __future__ import annotations
from dotenv import load_dotenv
import os
from pathlib import Path
import json
import re

import operator
from typing import TypedDict, List, Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str

    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section.",
    )
    bullets: List[str] = Field(
        ...,
        min_length=1,
        max_length=6,
        description="3–6 concrete, non-overlapping subpoints to cover in this section.",
    )
    target_words: int = Field(..., description="Target word count for this section (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  
    snippet: Optional[str] = None
    source: Optional[str] = None



class RouterDecision(BaseModel):
    needs_research: bool = Field(
        ...,
        description="true if web research is needed, otherwise false"
    )
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list) 


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)
    
class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  
    final: str

# -----------------------------
# 2) Worker State
# -----------------------------
class WorkerState(TypedDict):
    task: dict
    topic: str
    mode: str
    plan: dict
    evidence: List[dict]

    
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


# -----------------------------
# 3) Router (decide upfront)
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a blog generation system.

You analyze any blog topic including:
- technology
- health
- lifestyle
- fashion
- business
- education
- science

Decide whether external web research is needed before planning.

Return ONLY the structured output.

Important:
needs_research must be a JSON boolean.
Correct:
{
 "needs_research": false
}

Incorrect:
{
 "needs_research": "false"
}


Modes:
- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
- open_book (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.


If needs_research=true:
- Output 2-3 high-signal queries.
- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
- If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES.

You MUST always return all keys:

{
 "needs_research":,
 "mode": ,
 "queries": []
}

Never omit mode or queries

Important:
needs_research MUST be a JSON boolean — never a string.

CORRECT:   {"needs_research": false}
INCORRECT: {"needs_research": "false"}
INCORRECT: {"needs_research": "False"}

This applies to ALL topics: health, lifestyle, fashion, food, etc.
"""


def router_node(state: State) -> dict:
    topic = state["topic"]
    response = llm.invoke(
    [
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"Topic: {topic}\nAs-of date: {state.get('as_of')}"),
    ]
    )

    data = json.loads(response.content)

    if isinstance(data.get("needs_research"), str):
        data["needs_research"] = data["needs_research"].lower() == "true"

    decision = RouterDecision(**data)

    return {
        "needs_research": decision.needs_research, # Pass the safe value here
        "mode": decision.mode,
        "queries": decision.queries,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# -----------------------------
# 4) Research (Tavily) 
# -----------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query})

    normalized: List[dict] = []
    for r in results or []:
        normalized.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            }
        )
    return normalized


RESEARCH_SYSTEM = """You are a research synthesizer for blog writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Return the output as valid JSON.

The JSON must follow this exact structure:

{
  "evidence": [
    {
      "title": "string",
      "url": "string",
      "published_at": "YYYY-MM-DD or null",
      "snippet": "string or null",
      "source": "string or null"
    }
  ]
}

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do NOT guess.
- Keep snippets short.
- Deduplicate by URL.
"""

def research_node(state: State) -> dict:
    queries = (state.get("queries", []) or [])
    max_results = 2

    raw_results: List[dict] = []

    for q in queries:
        raw_results.extend(_tavily_search(q, max_results=max_results))

    if not raw_results:
        return {"evidence": []}
    
    compact_results = [
    {
        "title": r["title"],
        "url": r["url"],
        "snippet": (r.get("snippet") or "")[:300]
    }
    for r in raw_results
    ]

    extractor = llm.with_structured_output(EvidencePack, method="json_mode")
    pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(content=f"Raw results:\n{compact_results}"),
        ]
    )

    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e

    return {"evidence": list(dedup.values())}

# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior content writer who creates high-quality blogs 
across multiple domains including technology, health, lifestyle, fashion, business, and education.
Your job is to produce a highly actionable outline for a blog post.

Return ONLY valid JSON. Do not call tools. Do not add explanations.

CRITICAL: The JSON must match this EXACT schema structure, pay close attention to the keys:
{
 "blog_title": "string",
 "audience": "string",
 "tone": "string",
 "blog_kind": "explainer", 
 "constraints": ["string"],
 "tasks": [
   {
     "id": 1,
     "title": "string",
     "goal": "string",
     "bullets": ["string", "string", "string", "string"],
     "target_words": 300,
     "tags": ["string"],
     "requires_research": false,
     "requires_citations": false,
     "requires_code": false
   }
 ]
}

HARD KEY REQUIREMENTS (DO NOT FAIL THESE):
1. `blog_kind` MUST ONLY be one of: "explainer", "tutorial", "news_roundup", "comparison", or "system_design". DO NOT put "hybrid" or "open_book" here.
2. Every task inside "tasks" MUST have an integer `id` (1, 2, 3, etc.).
3. Every task inside "tasks" MUST have a string `title`.
4. The word count key MUST be exactly `target_words` (DO NOT USE target_word_count).

Quality bar:
- Create 5–9 tasks inside the "tasks" array.
- Bullets must be actionable: build/compare/measure/verify/debug.
- Ensure the overall plan includes at least 2 of these somewhere:
  * minimal code sketch / MWE (set requires_code=true for that section)
  * edge cases / failure modes
  * performance/cost considerations

Grounding rules:
- Mode closed_book: keep it evergreen; do not depend on evidence.
- Mode hybrid: Use evidence for up-to-date examples (models/tools). Mark sections using fresh info as requires_research=true.
- Mode open_book: Set blog_kind = "news_roundup".
"""

def orchestrator_node(state: State) -> dict:
    planner = llm.with_structured_output(Plan, method="json_mode")

    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n\n"
                    f"Evidence (ONLY use for fresh claims; may be empty):\n"
                    f"{[e.model_dump() for e in evidence][:16]}"
                )
            ),
        ]
    )

    return {"plan": plan}

# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]
    
# -----------------------------
# 7) Worker (write one section)
# -----------------------------
WORKER_SYSTEM = """You are a senior content writer who creates high-quality blogs 
across multiple domains including technology, health, lifestyle, fashion, business, and education.
Write ONE section of a blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
- Start with a '## <Section Title>' heading.
- LENGTH CONSTRAINT: You MUST write highly detailed, in-depth paragraphs for each bullet point to hit the target word count. Do not write short summaries. Expand on concepts, provide examples, and explain the "why" and "how". 

Scope guard:
- If blog_kind == "news_roundup": Focus on summarizing events and implications.

Grounding policy:
- If mode == open_book: Only use URLs provided in Evidence. Attach a source as a Markdown link: ([Source](URL)).
- If requires_citations == true: Cite Evidence URLs.

Code:
- If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.

Style:
- Detailed paragraphs, bullets where helpful, code fences for code.
- Avoid fluff/marketing. Be precise and implementation-oriented.
"""

def worker_node(state: WorkerState) -> dict: 
    
    task = Task(**state["task"])
    plan = Plan(**state["plan"])
    evidence = [EvidenceItem(**e) for e in state.get("evidence", [])]
    topic = state["topic"]
    mode = state.get("mode", "closed_book")

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
            for e in evidence[:20]
        )

    section_md = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words} (CRITICAL: You must generate AT LEAST this many words. Write deeply and comprehensively.)\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}

# -----------------------------
# 8) Reducer (merge + save)
# -----------------------------
def reducer_node(state: State) -> dict:

    plan = state["plan"]

    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    final_md = f"# {plan.blog_title}\n\n{body}\n"

    safe_title = "".join(c if c.isalnum() else "_" for c in plan.blog_title)
    filename = f"{safe_title}.md"
    Path(filename).write_text(final_md, encoding="utf-8")

    return {"final": final_md}

# -----------------------------
# 9) Build graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_node)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()

# -----------------------------
# 10) Runner
# -----------------------------
def run(topic: str):
    out = app.invoke(
        {
            "topic": topic
        }
        ,config={"max_concurrency": 1}
    )
    return out

if __name__ == "__main__":
    res = run("health")
    print("Blog Generated Successfully!")
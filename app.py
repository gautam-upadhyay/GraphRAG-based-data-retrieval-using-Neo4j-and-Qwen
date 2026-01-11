import os
import json
import re
from typing import List, Dict, Any

import requests
from neo4j import GraphDatabase

from intent import classify_intent_and_entities
from cypher_builder import build_queries
from ranker import score_items, select_top_n, lexical_overlap_score


def get_env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


NEO4J_URI = get_env("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = get_env("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = get_env("NEO4J_PASSWORD", "12345678")
NEO4J_DATABASE = get_env("NEO4J_DATABASE", "neo4j")

OLLAMA_API_URL = get_env("OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = get_env("OLLAMA_MODEL", "qwen2:7b")


class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def run(self, query: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        with self.driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]


def sanitize_question(question: str) -> str:
    return question.strip()


STOPWORDS = {
    "what","is","are","the","of","and","for","a","an","in","to","on","with",
    "from","this","that","it","does","do","does","how","which","who","where","when",
}


def extract_keywords(question: str, max_keywords: int = 5) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", question.lower())
    seen = set()
    keywords: List[str] = []
    for w in words:
        if w in STOPWORDS:
            continue
        if w not in seen:
            seen.add(w)
            keywords.append(w)
        if len(keywords) >= max_keywords:
            break
    return keywords


def retrieve_graph_context(db: Neo4jClient, question: str, limit: int = 12) -> Dict[str, Any]:
    q = sanitize_question(question)
    keywords = extract_keywords(q)
    key = keywords[0] if keywords else q

    query_sections = (
        """
        MATCH (s:Section)
        WHERE toLower(s.text) CONTAINS toLower($q)
        WITH s LIMIT $limit
        OPTIONAL MATCH (s)-[r]-(n)
        WITH s, collect(distinct {rel:type(r), otherLabels:labels(n), otherName: coalesce(n.name, n.title, n.full_name)}) AS rels
        RETURN collect(distinct {
          labels: labels(s),
          name: s.name,
          text: s.text,
          rels: rels
        }) AS sections
        """
    )

    query_entities = (
        """
        CALL {
          WITH $key AS k
          MATCH (a:Algorithm)
          WHERE toLower(coalesce(a.name, "")) CONTAINS toLower(k)
             OR toLower(coalesce(a.full_name, "")) CONTAINS toLower(k)
          RETURN a AS e
          UNION
          WITH $key AS k
          MATCH (c:CancerType)
          WHERE toLower(coalesce(c.name, "")) CONTAINS toLower(k)
          RETURN c AS e
        }
        WITH e LIMIT $limit
        OPTIONAL MATCH (e)-[r]-(n)
        WITH e, collect(distinct {rel:type(r), otherLabels:labels(n), otherName: coalesce(n.name, n.title, n.full_name)}) AS rels
        RETURN collect(distinct {
          labels: labels(e),
          name: coalesce(e.name, e.full_name, e.title),
          rels: rels
        }) AS entities
        """
    )

    query_results = (
        """
        MATCH (m:Model)-[:HAS_RESULT]->(r:Result)
        WHERE toLower($key) = ''
           OR toLower(m.name) CONTAINS toLower($key)
           OR toLower(coalesce(m.full_name, '')) CONTAINS toLower($key)
        RETURN collect({model: coalesce(m.full_name, m.name), metric: r.metric, accuracy: r.accuracy}) AS results
        """
    )

    query_best = (
        """
        OPTIONAL MATCH (:Paper)-[:BEST_MODEL]->(m:Model)
        RETURN collect({bestModel: coalesce(m.full_name, m.name)}) AS best
        """
    )

    query_intro_symptoms = (
        """
        MATCH (:Introduction)-[:MENTIONS_SYMPTOM]->(sym:Symptom)
        RETURN collect(distinct sym.name) AS symptoms
        """
    )

    query_intro_risks = (
        """
        MATCH (:Introduction)-[:IDENTIFIES_RISK_FACTOR]->(r:RiskFactor)
        RETURN collect(distinct r.name) AS risks
        """
    )

    query_intro_techniques = (
        """
        MATCH (:Introduction)-[:USES_TECHNIQUE]->(t:Technique)
        RETURN collect(distinct t.name) AS techniques
        """
    )

    query_intro_cancer_types = (
        """
        MATCH (:Introduction)-[:DISCUSSES_CANCER_TYPE]->(c:CancerType)
        RETURN collect(distinct c.name) AS cancerTypes
        """
    )

    query_dataset = (
        """
        MATCH (:Methodology)-[:USES_DATASET]->(d:Dataset)
        RETURN d.name AS name, d.source AS source, d.instances AS instances, d.features AS features, d.format AS format
        """
    )

    query_conclusion = (
        """
        MATCH (s:Section:Conclusion)
        RETURN s.name AS name, s.text AS text
        """
    )

    ctx: Dict[str, Any] = {
        "sections": [],
        "entities": [],
        "results": [],
        "best": [],
        "symptoms": [],
        "risks": [],
        "techniques": [],
        "cancerTypes": [],
        "dataset": {},
        "conclusion": "",
    }

    res1 = db.run(query_sections, {"q": q, "limit": limit})
    if res1:
        ctx["sections"] = res1[0].get("sections", [])

    res2 = db.run(query_entities, {"key": key, "limit": limit})
    if res2:
        ctx["entities"] = res2[0].get("entities", [])

    res3 = db.run(query_results, {"key": key})
    if res3:
        ctx["results"] = res3[0].get("results", [])

    res4 = db.run(query_best, {})
    if res4:
        ctx["best"] = res4[0].get("best", [])

    r_sym = db.run(query_intro_symptoms, {})
    if r_sym:
        ctx["symptoms"] = r_sym[0].get("symptoms", [])

    r_risk = db.run(query_intro_risks, {})
    if r_risk:
        ctx["risks"] = r_risk[0].get("risks", [])

    r_tech = db.run(query_intro_techniques, {})
    if r_tech:
        ctx["techniques"] = r_tech[0].get("techniques", [])

    r_ct = db.run(query_intro_cancer_types, {})
    if r_ct:
        ctx["cancerTypes"] = r_ct[0].get("cancerTypes", [])

    r_ds = db.run(query_dataset, {})
    if r_ds:
        ctx["dataset"] = r_ds[0]

    r_conc = db.run(query_conclusion, {})
    if r_conc and r_conc[0].get("text"):
        ctx["conclusion"] = r_conc[0]["text"]

    return ctx


def format_context(ctx: Dict[str, Any], max_chars: int = 4000) -> str:
    lines: List[str] = []
    sections = ctx.get("sections", [])
    entities = ctx.get("entities", [])

    if sections:
        lines.append("[Sections]")
        for s in sections[:8]:
            text = (s.get("text") or "")
            name = (s.get("name") or "Section")
            lines.append(f"- {name}: {text[:300].replace('\n',' ')}")

    if entities:
        lines.append("[Entities]")
        for e in entities[:8]:
            labels = ":".join(e.get("labels", []))
            name = e.get("name") or "(unnamed)"
            lines.append(f"- {labels} | {name}")

    results = ctx.get("results", [])
    if results:
        lines.append("[Results]")
        for r in results[:8]:
            model = r.get("model")
            metric = r.get("metric")
            acc = r.get("accuracy")
            lines.append(f"- {model}: {metric} = {acc}")

    best = ctx.get("best", [])
    if best:
        for b in best:
            if b.get("bestModel"):
                lines.append(f"[BestModel] {b.get('bestModel')}")

    if ctx.get("symptoms"):
        lines.append("[Symptoms]")
        lines.append(", ".join(sorted(set(ctx["symptoms"]))[:30]))
    if ctx.get("risks"):
        lines.append("[RiskFactors]")
        lines.append(", ".join(sorted(set(ctx["risks"]))[:30]))
    if ctx.get("techniques"):
        lines.append("[DiagnosticTechniques]")
        lines.append(", ".join(sorted(set(ctx["techniques"]))[:30]))
    if ctx.get("cancerTypes"):
        lines.append("[CancerTypes]")
        lines.append(", ".join(sorted(set(ctx["cancerTypes"]))[:30]))

    ds = ctx.get("dataset") or {}
    if ds:
        lines.append("[Dataset]")
        parts = []
        for k in ["name","source","instances","features","format"]:
            if ds.get(k) is not None:
                parts.append(f"{k}={ds.get(k)}")
        if parts:
            lines.append("; ".join(parts))

    if ctx.get("conclusion"):
        snippet = (ctx["conclusion"] or "")[:400].replace("\n", " ")
        lines.append("[Conclusion]")
        lines.append(snippet)

    context_str = "\n".join(lines)
    if len(context_str) > max_chars:
        context_str = context_str[: max_chars - 100] + "\n... [truncated]"
    return context_str


def build_prompt(question: str, context: str) -> str:
    return (
        "You are a domain expert answering strictly from the provided Neo4j graph context.\n"
        "If the context is insufficient or unrelated, answer exactly: 'I don't know from the graph.'\n"
        "When you answer, ground each claim by referencing the bracketed sections (e.g., [Symptoms], [Results], [Dataset], [Conclusion]). Do not invent facts.\n\n"
        f"Question: {question}\n\n"
        f"Graph Context:\n{context}\n\n"
        "Answer concisely in 3-6 sentences."
    )


def validate_answer(answer_text: str, context_text: str) -> str:
    used_tags = re.findall(r"\[(Symptoms|RiskFactors|DiagnosticTechniques|Dataset|CancerTypes|Results|BestModel|Sections|Conclusion)\]", answer_text)
    available_tags = re.findall(r"\[(Symptoms|RiskFactors|DiagnosticTechniques|Dataset|CancerTypes|Results|BestModel|Sections|Conclusion)\]", context_text)
    if available_tags and not used_tags:
        return "[Sections] " + answer_text
    return answer_text


def call_ollama(prompt: str) -> str:
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0, "top_p": 0.1, "repeat_penalty": 1.1}}
    response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    # Ollama returns { response: "...", done: true, ... }
    return data.get("response", "")


def answer(question: str) -> str:
    # 1) classify
    classification = classify_intent_and_entities(question)
    intent = classification["intent"]
    entities = classification["entities"]

    # 2) build queries
    specs = build_queries(intent=intent, entities=entities, question_text=question)

    db = Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        all_scored: List[Dict[str, Any]] = []
        context_lines: List[str] = []
        for spec in specs:
            rows = db.run(spec["query"], spec.get("params") or {})
            scored = score_items(question, spec["tag"], rows, entities)
            top = select_top_n(scored, n=8)
            if top:
                context_lines.append(f"[{spec['tag']}]")
                for r in top:
                    if spec["tag"] in {"Symptoms","RiskFactors","DiagnosticTechniques","CancerTypes"} and r.get("item"):
                        context_lines.append(f"- {r['item']}")
                    elif spec["tag"] == "Dataset":
                        parts = []
                        for k in ["name","source","instances","features","format"]:
                            if r.get(k) is not None:
                                parts.append(f"{k}={r.get(k)}")
                        if parts:
                            context_lines.append("; ".join(parts))
                    elif spec["tag"] == "Results":
                        context_lines.append(f"- {r.get('model')}: {r.get('metric')} = {r.get('accuracy')}")
                    elif spec["tag"] == "BestModel":
                        if r.get('bestModel'):
                            context_lines.append(f"- {r.get('bestModel')}")
                    elif spec["tag"] in {"Sections","Conclusion"}:
                        name = r.get('name') or spec["tag"]
                        text = (r.get('text') or "").replace('\n',' ')[:400]
                        context_lines.append(f"- {name}: {text}")
                all_scored.extend(top)
    finally:
        db.close()

    # 6) build final context string
    context_str = "\n".join(context_lines)
    if not context_str.strip():
        return "I don't know from the graph."

    prompt = build_prompt(question, context_str)
    raw_answer = call_ollama(prompt)
    return validate_answer(raw_answer, context_str)


if __name__ == "__main__":
    import argparse

    

    if not args.question:
        print("Usage: python app.py -q \"What symptoms indicate lung cancer?\"")
        raise SystemExit(2)

    # Utility to compute and print Section + Score table
    def print_section_scores(question: str) -> None:
        db = Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        try:
            classification = classify_intent_and_entities(question)
            intent = classification["intent"]

            # Fetch candidate sections' text
            section_queries = [
                ("Introduction", "MATCH (s:Section:Introduction) RETURN s.name AS name, s.text AS text"),
                ("Methodology", "MATCH (s:Section:Methodology) RETURN s.name AS name, s.text AS text"),
                ("Results", "MATCH (s:Section:Results) RETURN s.name AS name, s.text AS text"),
                ("Conclusion", "MATCH (s:Section:Conclusion) RETURN s.name AS name, s.text AS text"),
            ]

            sections = []
            for label, q in section_queries:
                rows = db.run(q)
                if rows:
                    r = rows[0]
                    sections.append({"label": label, "name": r.get("name"), "text": r.get("text") or ""})

            # Graph counts per section by intent
            counts = {"Introduction": 0, "Methodology": 0, "Results": 0, "Conclusion": 0}
            if intent == "symptoms":
                r = db.run("MATCH (:Introduction)-[:MENTIONS_SYMPTOM]->(s) RETURN count(s) AS c")
                counts["Introduction"] = int(r[0]["c"]) if r else 0
                r = db.run("MATCH (:Dataset)-[:HAS_FEATURE]->(:Feature:Symptom) RETURN count(*) AS c")
                counts["Methodology"] = int(r[0]["c"]) if r else 0
            elif intent == "risk_factors":
                r = db.run("MATCH (:Introduction)-[:IDENTIFIES_RISK_FACTOR]->(r) RETURN count(r) AS c")
                counts["Introduction"] = int(r[0]["c"]) if r else 0
            elif intent == "diagnostic_techniques":
                r = db.run("MATCH (:Introduction)-[:USES_TECHNIQUE]->(t) RETURN count(t) AS c")
                counts["Introduction"] = int(r[0]["c"]) if r else 0
            elif intent == "cancer_types":
                r = db.run("MATCH (:Introduction)-[:DISCUSSES_CANCER_TYPE]->(c) RETURN count(c) AS c")
                counts["Introduction"] = int(r[0]["c"]) if r else 0
            elif intent == "dataset":
                r = db.run("MATCH (:Methodology)-[:USES_DATASET]->(d) RETURN count(d) AS c")
                counts["Methodology"] = int(r[0]["c"]) if r else 0
            elif intent == "results":
                r = db.run("MATCH (:Results)-[:CONTAINS_RESULT]->(:Result) RETURN count(*) AS c")
                counts["Results"] = int(r[0]["c"]) if r else 0
            elif intent == "conclusion":
                r = db.run("MATCH (s:Section:Conclusion) RETURN count(s) AS c")
                counts["Conclusion"] = int(r[0]["c"]) if r else 0

            # Compute lexical score and simple normalized graph score
            max_count = max(counts.values()) if counts else 0
            print("\n=== Section Scores ===\n")
            print(f"Question: {question}")
            print(f"Intent: {intent}\n")
            rows_out = []
            for s in sections:
                lex = lexical_overlap_score(question, s.get("text") or "")
                g = (counts.get(s["label"], 0) / max_count) if max_count else 0.0
                final = round(0.5 * lex + 0.5 * g, 4)
                rows_out.append((s["label"], round(lex, 4), round(g, 4), final))
            # Sort by final desc
            rows_out.sort(key=lambda x: x[3], reverse=True)
            # Print table
            print("Section\tLexical\tGraph\tFinal")
            for label, lex, g, final in rows_out:
                print(f"{label}\t{lex}\t{g}\t{final}")

        finally:
            db.close()

    try:
        if args.scores:
            print_section_scores(args.question)
        else:
            result = answer(args.question)
            print("\n=== Answer ===\n")
            print(result)
    except Exception as e:
        print(f"Error: {e}")



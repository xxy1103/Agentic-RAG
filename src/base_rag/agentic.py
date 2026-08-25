from __future__ import annotations

import json
import re
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from base_rag.bm25 import BM25Store
from base_rag.config import AppConfig
from base_rag.models import (
    AgenticRetrievalResult,
    EvidenceDecision,
    EvidenceRequirement,
    HopTrace,
    RequirementAssessment,
    RouteDecision,
    SearchHit,
)
from base_rag.pipeline import (
    Embedder,
    Generator,
    PipelineStageError,
    StageProgressCallback,
    _citation,
    _run_stage,
    retrieve_hits,
)
from base_rag.rerank import Reranker
from base_rag.store import FaissStore


def interleave_hop_hits(accumulated_hits_by_hop: list[list[SearchHit]], top_k: int) -> list[SearchHit]:
    """Round-robin interleave hits from each hop while deduplicating by chunk_id."""
    if not accumulated_hits_by_hop:
        return []
    seen_chunk_ids: set[str] = set()
    interleaved: list[SearchHit] = []
    max_len = max((len(hop_hits) for hop_hits in accumulated_hits_by_hop), default=0)
    for rank_idx in range(max_len):
        for hop_hits in accumulated_hits_by_hop:
            if rank_idx < len(hop_hits):
                hit = hop_hits[rank_idx]
                if hit.chunk.chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(hit.chunk.chunk_id)
                    interleaved.append(hit)
                    if len(interleaved) >= top_k:
                        break
        if len(interleaved) >= top_k:
            break

    final_hits: list[SearchHit] = []
    for rank, hit in enumerate(interleaved, start=1):
        final_hits.append(replace(hit, rank=rank))
    return final_hits


def _extract_json_block(text: str) -> str:
    cleaned = text.strip()
    match = re.search(r"```(?:json)?s*([sS]*?)```", cleaned, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return cleaned[first_brace : last_brace + 1].strip()
    return cleaned


def _call_structured_llm(
    generator: Generator,
    prompt: str,
    stage_name: str,
    validator: Callable[[dict[str, Any]], Any],
    max_retries: int = 1,
    temperature: float = 0.0,
    max_tokens: int = 600,
) -> tuple[Any, int]:
    """Call LLM and parse JSON output, retrying with error feedback if malformed."""
    attempts = 0
    current_prompt = prompt
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        attempts += 1
        raw_text = generator.generate(current_prompt, temperature=temperature, max_tokens=max_tokens)
        try:
            json_str = _extract_json_block(raw_text)
            data = json.loads(json_str)
            if not isinstance(data, dict):
                raise ValueError("输出内容必须是 JSON 对象。")
            validated = validator(data)
            return validated, attempts
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                current_prompt = (
                    f"{prompt}\n\n"
                    f"【注意】：前一次输出格式有误（{exc}）。请严格仅输出符合要求的 JSON 对象，不要包含其他解释文本。"
                )
    raise PipelineStageError(stage_name, last_error or RuntimeError("结构化输出解析失败"))


def route_query_llm(
    generator: Generator,
    question: str,
    max_retries: int = 1,
) -> tuple[RouteDecision, int]:
    prompt = f"""你是一个多跳检索路由助手（Query Router）。
请分析用户问题是否需要分解为多个检索步骤（多跳检索，发现中间实体/线索后才能找到最终答案），还是可以通过单次检索直接找到答案。

规则：
1. 如果是简单事实、单实体或单个主题问题，判定为 single_hop，query 保持或轻微优化原始问题。
2. 如果问题涉及实体跳转（如“A 创立者的毕业院校”）、对比多方事实或时间线推理，判定为 multi_hop，query 为寻找第一个中间实体的首跳检索词。
3. 将原问题拆成完整回答时必须由检索证据直接支持的最小需求清单。不要填写答案，不要凭常识补充问题中没有的条件。
4. single_hop 通常只有 1 项需求；multi_hop 应包含多个彼此可独立核验的需求。需求数不等于检索跳数：一次检索可以覆盖多项需求。
5. 不要把最终答案综合、比较结论或“确认前述实体相同”单独列为需求；这些应由前述原子证据自然推出。

请严格仅输出以下 JSON 格式：
{{
  "route": "single_hop" | "multi_hop",
  "query": "首步检索查询词",
  "reason": "判断依据",
  "requirements": [
    {{"requirement_id": "R1", "description": "必须检索并证明的条件"}}
  ]
}}

问题：{question}"""

    def _validate(data: dict[str, Any]) -> RouteDecision:
        route = str(data.get("route", "")).strip().lower()
        if route not in {"single_hop", "multi_hop"}:
            raise ValueError("route 必须是 single_hop 或 multi_hop。")
        query = str(data.get("query", "")).strip()
        reason = str(data.get("reason", "")).strip()
        if not query or not reason:
            raise ValueError("query 和 reason 不能为空。")
        requirements_raw = data.get("requirements")
        if not isinstance(requirements_raw, list) or not requirements_raw:
            raise ValueError("requirements 必须是非空数组。")
        requirements: list[EvidenceRequirement] = []
        seen_ids: set[str] = set()
        for item in requirements_raw:
            if not isinstance(item, dict):
                raise ValueError("每项 requirement 必须是 JSON 对象。")
            requirement_id = str(item.get("requirement_id", "")).strip()
            description = str(item.get("description", "")).strip()
            if not requirement_id or not description or requirement_id in seen_ids:
                raise ValueError("requirement_id 必须唯一，且 description 不能为空。")
            seen_ids.add(requirement_id)
            requirements.append(EvidenceRequirement(requirement_id, description))
        if route == "multi_hop" and len(requirements) < 2:
            raise ValueError("multi_hop 至少需要两项证据需求。")
        return RouteDecision(route=route, query=query, reason=reason, requirements=requirements)

    return _call_structured_llm(generator, prompt, "router", _validate, max_retries=max_retries, max_tokens=900)


def grade_evidence_llm(
    generator: Generator,
    question: str,
    current_query: str,
    requirements: list[EvidenceRequirement],
    hits: list[SearchHit],
    max_retries: int = 1,
) -> tuple[EvidenceDecision, int]:
    requirements_str = "\n".join(
        f'- {requirement.requirement_id}: {requirement.description}' for requirement in requirements
    )
    passages = []
    for hit in hits:
        passages.append(f"[chunk_id={hit.chunk.chunk_id}; {_citation(hit)}]\n{hit.chunk.text}")
    passages_str = "\n\n".join(passages) if passages else "（未检索到任何相关段落）"

    prompt = f"""你是一个多跳检索证据评估助手（Evidence Grader）。
请根据累计检索证据逐项评估证据需求。你的任务是判断证据集合是否完整，不是猜测最终答案。

原始问题：{question}
当前步查询词：{current_query}

必须覆盖的证据需求：
{requirements_str}

截至当前累计检索到的证据段落：
{passages_str}

强制规则：
1. 必须为每一项 requirement 输出一项 assessment，不能遗漏或增加需求。
2. 只有段落直接支持需求时才标记 supported，并绑定一个或多个真实存在的 chunk_id；否则标记 missing。
3. 已经知道或可以猜出最终答案，不代表证据完整。只要有任何 requirement 为 missing，就不得返回 complete。
4. 有缺失需求且本轮取得了有效进展时返回 continue，并针对一个缺失需求生成下一跳查询。
5. 本轮没有取得可靠进展时返回 insufficient，并说明失败原因。

请严格仅输出以下 JSON 格式：
{{
  "verdict": "complete" | "continue" | "insufficient",
  "requirement_assessments": [
    {{"requirement_id": "R1", "status": "supported" | "missing", "evidence_chunk_ids": ["真实 chunk_id"]}}
  ],
  "next_requirement_id": "下一步要解决的缺失需求 ID（若 continue）或 null",
  "next_query": "下一跳具体查询词（若 continue）或 null",
  "reason": "判断简要理由",
  "failure_reason": "证据不足的原因（若 insufficient）或 null"
}}"""

    def _validate(data: dict[str, Any]) -> EvidenceDecision:
        verdict = str(data.get("verdict", "")).strip().lower()
        if verdict not in {"complete", "continue", "insufficient"}:
            raise ValueError("verdict 必须是 complete、continue 或 insufficient。")
        expected_ids = [requirement.requirement_id for requirement in requirements]
        assessments_raw = data.get("requirement_assessments")
        if not isinstance(assessments_raw, list):
            raise ValueError("requirement_assessments 必须是数组。")
        assessments: list[RequirementAssessment] = []
        assessment_ids: list[str] = []
        for item in assessments_raw:
            if not isinstance(item, dict):
                raise ValueError("每项 requirement assessment 必须是 JSON 对象。")
            requirement_id = str(item.get("requirement_id", "")).strip()
            status = str(item.get("status", "")).strip().lower()
            chunk_ids_raw = item.get("evidence_chunk_ids", [])
            if status not in {"supported", "missing"} or not isinstance(chunk_ids_raw, list):
                raise ValueError("assessment status 或 evidence_chunk_ids 非法。")
            chunk_ids = list(dict.fromkeys(str(chunk_id).strip() for chunk_id in chunk_ids_raw if str(chunk_id).strip()))
            assessments.append(RequirementAssessment(requirement_id, status, chunk_ids))
            assessment_ids.append(requirement_id)
        if len(assessment_ids) != len(set(assessment_ids)) or set(assessment_ids) != set(expected_ids):
            raise ValueError("requirement_assessments 必须与 Router 的需求清单一一对应。")
        def _optional_string(value: Any) -> str | None:
            if value is None:
                return None
            normalized = str(value).strip()
            return None if not normalized or normalized.lower() in {"null", "none"} else normalized

        next_query = _optional_string(data.get("next_query"))
        next_requirement_id = _optional_string(data.get("next_requirement_id"))
        failure_reason = _optional_string(data.get("failure_reason"))
        reason = str(data.get("reason", "")).strip() or "证据评估"
        return EvidenceDecision(
            verdict=verdict,
            model_verdict=verdict,
            reason=reason,
            next_query=next_query,
            next_requirement_id=next_requirement_id,
            failure_reason=failure_reason,
            requirement_assessments=assessments,
        )

    return _call_structured_llm(generator, prompt, "grader", _validate, max_retries=max_retries, max_tokens=1000)


def apply_complete_gate(
    decision: EvidenceDecision,
    requirements: list[EvidenceRequirement],
    hits: list[SearchHit],
) -> EvidenceDecision:
    """Ground requirement citations and make completeness a deterministic decision."""
    valid_chunk_ids = {hit.chunk.chunk_id for hit in hits}
    assessment_by_id = {assessment.requirement_id: assessment for assessment in decision.requirement_assessments}
    grounded: list[RequirementAssessment] = []
    for requirement in requirements:
        assessment = assessment_by_id.get(requirement.requirement_id)
        cited = [chunk_id for chunk_id in (assessment.evidence_chunk_ids if assessment else []) if chunk_id in valid_chunk_ids]
        status = "supported" if assessment and assessment.status == "supported" and cited else "missing"
        grounded.append(RequirementAssessment(requirement.requirement_id, status, cited))

    all_supported = bool(requirements) and all(item.status == "supported" for item in grounded)
    effective_verdict = "complete" if all_supported else decision.verdict
    missing = [item.requirement_id for item in grounded if item.status == "missing"]
    next_requirement_id = decision.next_requirement_id if decision.next_requirement_id in missing else (missing[0] if missing else None)
    next_query = decision.next_query
    reason = decision.reason
    if not all_supported and decision.verdict == "complete":
        effective_verdict = "continue"
        missing_requirement = next((item for item in requirements if item.requirement_id == next_requirement_id), None)
        if not next_query and missing_requirement:
            next_query = missing_requirement.description
        reason = f"complete 门控未通过；仍缺少需求：{', '.join(missing)}。{reason}"
    if effective_verdict == "continue" and not next_query:
        missing_requirement = next((item for item in requirements if item.requirement_id == next_requirement_id), None)
        next_query = missing_requirement.description if missing_requirement else None

    return replace(
        decision,
        verdict=effective_verdict,
        model_verdict=decision.model_verdict or decision.verdict,
        reason=reason,
        next_query=next_query,
        next_requirement_id=next_requirement_id,
        requirement_assessments=grounded,
    )


def correct_query_llm(
    generator: Generator,
    question: str,
    failed_query: str,
    failure_reason: str | None,
    missing_requirements: list[EvidenceRequirement],
    max_retries: int = 1,
) -> tuple[str, int]:
    missing_requirements_str = "\n".join(
        f"- {requirement.requirement_id}: {requirement.description}" for requirement in missing_requirements
    ) or "（未提供；请围绕失败原因改写）"
    prompt = f"""你是一个检索词纠错助手（Query Corrector）。
当前检索词未能检索到足够证据，请根据失败原因和未满足的证据需求，重新改写生成更具针对性、同义替换或关键词更精准的检索词。

原始问题：{question}
失败查询词：{failed_query}
失败原因：{failure_reason or '未检索到有效事实'}
未满足的证据需求：
{missing_requirements_str}

请严格仅输出以下 JSON 格式：
{{
  "corrected_query": "改写后的新检索词",
  "reason": "改写理由"
}}"""

    def _validate(data: dict[str, Any]) -> str:
        corrected = str(data.get("corrected_query", "")).strip()
        if not corrected:
            raise ValueError("缺少 corrected_query 字段。")
        return corrected

    return _call_structured_llm(generator, prompt, "corrector", _validate, max_retries=max_retries)


class AgenticState(TypedDict, total=False):
    question: str
    route_decision: RouteDecision | None
    current_hop: int
    current_query: str
    is_correction: bool
    correction_count_current_hop: int
    accumulated_hits_by_hop: list[list[SearchHit]]
    requirement_assessments: list[RequirementAssessment]
    traces: list[HopTrace]
    latest_hits: list[SearchHit]
    latest_decision: EvidenceDecision | None
    termination_reason: str
    final_hits: list[SearchHit]
    stage_calls: dict[str, int]
    stage_seconds: dict[str, float]
    error: str | None


def run_agentic_retrieval(
    config: AppConfig,
    embedder: Embedder,
    generator: Generator,
    question: str,
    *,
    reranker: Reranker | None = None,
    dense_store: FaissStore | None = None,
    bm25_store: BM25Store | None = None,
    on_stage: StageProgressCallback | None = None,
    run_log_dir: Path | None = None,
) -> AgenticRetrievalResult:
    """Execute bounded multi-hop Agentic retrieval orchestrated by LangGraph."""
    started = time.perf_counter()
    stage_calls = {
        "embedding": 0,
        "rerank": 0,
        "rewrite": 0,
        "router": 0,
        "grader": 0,
        "corrector": 0,
    }
    stage_seconds: dict[str, float] = {}

    if dense_store is None:
        dense_store = _run_stage(
            "index_load",
            lambda: FaissStore.load(config.paths.index_dir, config.models.embedding_model, config.models.embedding_dimensions),
            on_stage,
        )
    if bm25_store is None:
        bm25_store = _run_stage(
            "bm25_load",
            lambda: BM25Store.load(config.paths.index_dir, dense_store.chunks, dense_store.metadata.corpus_hash),
            on_stage,
        )

    builder = StateGraph(AgenticState)

    def node_route(state: AgenticState) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            decision, calls = _run_stage(
                "agentic_route",
                lambda: route_query_llm(generator, state["question"], max_retries=config.agentic.structured_output_retries),
                on_stage,
            )
            stage_calls["router"] += calls
            stage_seconds["router"] = round(time.perf_counter() - t0, 3)
            return {
                "route_decision": decision,
                "current_query": decision.query,
                "current_hop": 1,
                "is_correction": False,
                "correction_count_current_hop": 0,
            }
        except Exception as exc:
            stage_seconds["router"] = round(time.perf_counter() - t0, 3)
            fallback_route = RouteDecision(route="single_hop", query=state["question"], reason=f"路由失败: {exc}")
            return {
                "route_decision": fallback_route,
                "current_query": state["question"],
                "current_hop": 1,
                "is_correction": False,
                "correction_count_current_hop": 0,
                "termination_reason": "router_failed",
                "error": str(exc),
            }

    def node_retrieve(state: AgenticState) -> dict[str, Any]:
        t0 = time.perf_counter()
        query = state.get("current_query", state["question"])
        try:
            retrieval_res = retrieve_hits(
                config=config,
                embedder=embedder,
                question=query,
                profile=config.agentic.base_profile,
                reranker=reranker,
                generator=generator,
                dense_store=dense_store,
                bm25_store=bm25_store,
                top_k=config.retrieval.top_k,
                on_stage=on_stage,
            )
            hits: list[SearchHit] = retrieval_res["final_hits"]  # type: ignore
            sub_calls = retrieval_res.get("stage_calls", {})  # type: ignore
            for k_call, v_call in sub_calls.items():
                stage_calls[k_call] = stage_calls.get(k_call, 0) + v_call
            hop_hits_list = [list(h) for h in state.get("accumulated_hits_by_hop", [])]
            if state.get("is_correction", False) and hop_hits_list:
                hop_hits_list[-1] = hits
            else:
                hop_hits_list.append(hits)
            return {
                "latest_hits": hits,
                "accumulated_hits_by_hop": hop_hits_list,
            }
        except Exception as exc:
            return {
                "termination_reason": "retrieval_failed",
                "error": str(exc),
            }

    def node_grade(state: AgenticState) -> dict[str, Any]:
        t0 = time.perf_counter()
        hits = state.get("latest_hits", [])
        question_text = state["question"]
        current_q = state.get("current_query", question_text)
        route_decision = state.get("route_decision")
        requirements = route_decision.requirements if route_decision else []
        accumulated_by_hop = state.get("accumulated_hits_by_hop", [])
        accumulated_count = sum(len(hop_hits) for hop_hits in accumulated_by_hop)
        accumulated_hits = interleave_hop_hits(accumulated_by_hop, accumulated_count) if accumulated_count else []
        try:
            decision, calls = _run_stage(
                "agentic_grade",
                lambda: grade_evidence_llm(
                    generator=generator,
                    question=question_text,
                    current_query=current_q,
                    requirements=requirements,
                    hits=accumulated_hits,
                    max_retries=config.agentic.structured_output_retries,
                ),
                on_stage,
            )
            decision = apply_complete_gate(decision, requirements, accumulated_hits)
            stage_calls["grader"] += calls
            stage_seconds["grader"] = stage_seconds.get("grader", 0.0) + round(time.perf_counter() - t0, 3)
            trace = HopTrace(
                hop_index=state.get("current_hop", 1),
                is_correction=state.get("is_correction", False),
                query=current_q,
                hits=hits,
                decision=decision,
                elapsed_seconds=round(time.perf_counter() - t0, 3),
                stage_calls=dict(stage_calls),
            )
            traces = list(state.get("traces", []))
            traces.append(trace)
            return {
                "latest_decision": decision,
                "requirement_assessments": decision.requirement_assessments,
                "traces": traces,
            }
        except Exception as exc:
            stage_seconds["grader"] = stage_seconds.get("grader", 0.0) + round(time.perf_counter() - t0, 3)
            trace = HopTrace(
                hop_index=state.get("current_hop", 1),
                is_correction=state.get("is_correction", False),
                query=current_q,
                hits=hits,
                decision=None,
                elapsed_seconds=round(time.perf_counter() - t0, 3),
                stage_calls=dict(stage_calls),
            )
            traces = list(state.get("traces", []))
            traces.append(trace)
            return {
                "traces": traces,
                "termination_reason": "grader_failed",
                "error": str(exc),
            }

    def node_prepare_next_hop(state: AgenticState) -> dict[str, Any]:
        decision = state.get("latest_decision")
        next_q = decision.next_query if decision and decision.next_query else state["question"]
        return {
            "current_hop": state.get("current_hop", 1) + 1,
            "current_query": next_q,
            "is_correction": False,
            "correction_count_current_hop": 0,
        }

    def node_correct_query(state: AgenticState) -> dict[str, Any]:
        t0 = time.perf_counter()
        question_text = state["question"]
        current_q = state.get("current_query", question_text)
        decision = state.get("latest_decision")
        failure_reason = decision.failure_reason if decision else None
        route_decision = state.get("route_decision")
        requirements = route_decision.requirements if route_decision else []
        assessment_by_id = {
            assessment.requirement_id: assessment for assessment in state.get("requirement_assessments", [])
        }
        missing_requirements = [
            requirement for requirement in requirements
            if assessment_by_id.get(requirement.requirement_id) is None
            or assessment_by_id[requirement.requirement_id].status == "missing"
        ]
        try:
            corrected_q, calls = _run_stage(
                "agentic_correct",
                lambda: correct_query_llm(
                    generator=generator,
                    question=question_text,
                    failed_query=current_q,
                    failure_reason=failure_reason,
                    missing_requirements=missing_requirements,
                    max_retries=config.agentic.structured_output_retries,
                ),
                on_stage,
            )
            stage_calls["corrector"] += calls
            stage_seconds["corrector"] = stage_seconds.get("corrector", 0.0) + round(time.perf_counter() - t0, 3)
            return {
                "current_query": corrected_q,
                "is_correction": True,
                "correction_count_current_hop": state.get("correction_count_current_hop", 0) + 1,
            }
        except Exception:
            stage_seconds["corrector"] = stage_seconds.get("corrector", 0.0) + round(time.perf_counter() - t0, 3)
            return {
                "current_query": current_q,
                "is_correction": True,
                "correction_count_current_hop": state.get("correction_count_current_hop", 0) + 1,
            }

    def node_finalize(state: AgenticState) -> dict[str, Any]:
        accumulated = state.get("accumulated_hits_by_hop", [])
        final_hits = interleave_hop_hits(accumulated, config.retrieval.top_k)
        reason = state.get("termination_reason")
        if not reason:
            decision = state.get("latest_decision")
            if decision and decision.verdict == "complete":
                reason = "complete"
            elif state.get("current_hop", 1) >= config.agentic.max_hops:
                reason = "max_hops_reached"
            else:
                reason = "insufficient_evidence"
        return {
            "final_hits": final_hits,
            "termination_reason": reason,
        }

    builder.add_node("route_query", node_route)
    builder.add_node("retrieve_hop", node_retrieve)
    builder.add_node("grade_evidence", node_grade)
    builder.add_node("prepare_next_hop", node_prepare_next_hop)
    builder.add_node("correct_query", node_correct_query)
    builder.add_node("finalize", node_finalize)

    def route_after_route(state: AgenticState) -> str:
        if state.get("termination_reason") == "router_failed":
            return "finalize"
        return "retrieve_hop"

    def route_after_retrieve(state: AgenticState) -> str:
        if state.get("termination_reason") == "retrieval_failed":
            return "finalize"
        return "grade_evidence"

    def route_after_grade(state: AgenticState) -> str:
        if state.get("termination_reason") == "grader_failed":
            return "finalize"
        decision = state.get("latest_decision")
        if decision is None or decision.verdict == "complete":
            return "finalize"
        if decision.verdict == "continue":
            if state.get("current_hop", 1) >= config.agentic.max_hops:
                return "finalize"
            return "prepare_next_hop"
        if decision.verdict == "insufficient":
            if state.get("correction_count_current_hop", 0) >= config.agentic.max_corrections_per_hop:
                return "finalize"
            return "correct_query"
        return "finalize"

    builder.add_edge(START, "route_query")
    builder.add_conditional_edges("route_query", route_after_route, {"finalize": "finalize", "retrieve_hop": "retrieve_hop"})
    builder.add_conditional_edges("retrieve_hop", route_after_retrieve, {"finalize": "finalize", "grade_evidence": "grade_evidence"})
    builder.add_conditional_edges("grade_evidence", route_after_grade, {"finalize": "finalize", "prepare_next_hop": "prepare_next_hop", "correct_query": "correct_query"})
    builder.add_edge("prepare_next_hop", "retrieve_hop")
    builder.add_edge("correct_query", "retrieve_hop")
    builder.add_edge("finalize", END)

    compiled_graph = builder.compile()

    initial_state: AgenticState = {
        "question": question,
        "accumulated_hits_by_hop": [],
        "requirement_assessments": [],
        "traces": [],
        "stage_calls": stage_calls,
        "stage_seconds": stage_seconds,
    }

    try:
        final_state = compiled_graph.invoke(
            initial_state,
            config={"recursion_limit": config.agentic.recursion_limit},
        )
    except Exception as exc:
        final_state = initial_state
        final_state["termination_reason"] = "recursion_guard"
        final_state["error"] = str(exc)
        final_state["final_hits"] = interleave_hop_hits(final_state.get("accumulated_hits_by_hop", []), config.retrieval.top_k)

    elapsed_total = round(time.perf_counter() - started, 3)
    route_dec = final_state.get("route_decision") or RouteDecision("single_hop", question, "初始路由")
    traces_list = final_state.get("traces", [])
    final_hits = final_state.get("final_hits", [])
    termination_reason = final_state.get("termination_reason") or "complete"
    total_hops = max(1, len(set(t.hop_index for t in traces_list))) if traces_list else 1
    corrections_count = sum(1 for t in traces_list if t.is_correction)

    result = AgenticRetrievalResult(
        question=question,
        route_decision=route_dec,
        final_hits=final_hits,
        traces=traces_list,
        termination_reason=termination_reason,
        total_hops=total_hops,
        correction_count=corrections_count,
        elapsed_seconds=elapsed_total,
        stage_calls=stage_calls,
    )

    if config.runtime.save_runs:
        _save_agentic_run(run_log_dir or config.paths.runs_dir / "agentic", result, config.safe_dict())

    return result


def _save_agentic_run(log_dir: Path, result: AgenticRetrievalResult, config_dict: dict[str, object]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    payload = {
        "result": result.to_dict(),
        "config": config_dict,
    }
    (log_dir / f"{stamp}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

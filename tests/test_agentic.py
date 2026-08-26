"""Unit and integration tests for Phase 3 Agentic Multi-Hop Retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from base_rag.agentic import (
    _call_structured_llm,
    apply_complete_gate,
    compose_final_hits,
    correct_query_llm,
    grade_evidence_llm,
    interleave_hop_hits,
    route_query_llm,
    run_agentic_retrieval,
)
from base_rag.agentic_eval import (
    _calc_paired_deltas,
    _calc_system_metrics,
    get_or_create_multihop_splits,
    score_agentic_query,
)
from base_rag.bm25 import BM25Store
from base_rag.config import load_config
from base_rag.models import (
    AgenticRetrievalResult,
    Chunk,
    EvidenceDecision,
    EvidenceRequirement,
    RequirementAssessment,
    RouteDecision,
    SearchHit,
)
from base_rag.pipeline import PipelineStageError
from base_rag.store import FaissStore, IndexMetadata


def _chunk(identifier: str, text: str, source: str = 'doc1.md') -> Chunk:
    return Chunk(
        chunk_id=identifier,
        text=text,
        source_id=source,
        source_path=source,
        media_type='text/markdown',
        ordinal=0,
    )


def _hit(identifier: str, text: str, rank: int, score: float = 0.9, source: str = 'doc1.md') -> SearchHit:
    return SearchHit(chunk=_chunk(identifier, text, source), score=score, rank=rank)


def test_interleave_hop_hits_deduplicates_and_round_robins() -> None:
    hop1 = [_hit('c1', 'chunk 1', 1), _hit('c2', 'chunk 2', 2), _hit('c3', 'chunk 3', 3)]
    hop2 = [_hit('c2', 'chunk 2 dup', 1), _hit('c4', 'chunk 4', 2), _hit('c5', 'chunk 5', 3)]
    hop3 = [_hit('c6', 'chunk 6', 1)]

    interleaved = interleave_hop_hits([hop1, hop2, hop3], top_k=4)
    assert len(interleaved) == 4
    # Round 0: hop1[0] (c1), hop2[0] (c2), hop3[0] (c6)
    # Round 1: hop1[1] (c2 - skipped dup), hop2[1] (c4)
    ids = [h.chunk.chunk_id for h in interleaved]
    assert ids == ['c1', 'c2', 'c6', 'c4']
    assert [h.rank for h in interleaved] == [1, 2, 3, 4]


def test_compose_final_hits_prioritizes_requirement_bound_evidence_then_round_robins() -> None:
    hop1 = [_hit('h1-r1', 'hop one first', 1), _hit('bound-r1', 'R1 evidence', 2), _hit('h1-r3', 'hop one third', 3)]
    hop2 = [_hit('h2-r1', 'hop two first', 1), _hit('h2-r2', 'hop two second', 2), _hit('bound-r2', 'R2 evidence', 3)]
    assessments = [
        RequirementAssessment('R1', 'supported', ['bound-r1']),
        RequirementAssessment('R2', 'supported', ['bound-r2', 'missing-chunk']),
        RequirementAssessment('R3', 'missing', ['h1-r3']),
    ]

    final_hits = compose_final_hits([hop1, hop2], assessments, top_k=5)

    # Bound evidence is diversified by requirement before fallback; unsupported
    # assessments and nonexistent chunk IDs must not occupy final slots.
    assert [hit.chunk.chunk_id for hit in final_hits] == ['bound-r1', 'bound-r2', 'h1-r1', 'h2-r1', 'h2-r2']
    assert [hit.rank for hit in final_hits] == [1, 2, 3, 4, 5]


def test_compose_final_hits_without_bound_evidence_preserves_old_round_robin() -> None:
    hop1 = [_hit('c1', 'chunk 1', 1), _hit('c2', 'chunk 2', 2)]
    hop2 = [_hit('c3', 'chunk 3', 1), _hit('c4', 'chunk 4', 2)]
    assert [hit.chunk.chunk_id for hit in compose_final_hits([hop1, hop2], [], top_k=4)] == ['c1', 'c3', 'c2', 'c4']


class SequentialGenerator:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 600) -> str:
        self.calls.append(prompt)
        if not self.responses:
            raise RuntimeError('No more responses configured in FakeGenerator')
        return self.responses.pop(0)


def test_structured_llm_json_retry_on_malformed_text() -> None:
    generator = SequentialGenerator([
        'Here is the response: not valid json at all',
        '```json\n{"key": "success"}\n```',
    ])
    res, calls = _call_structured_llm(
        generator=generator,
        prompt='Please output JSON',
        stage_name='test_stage',
        validator=lambda d: d.get('key'),
        max_retries=1,
    )
    assert res == 'success'
    assert calls == 2


def test_structured_llm_fails_after_retries() -> None:
    generator = SequentialGenerator([
        'bad json 1',
        'bad json 2',
    ])
    with pytest.raises(PipelineStageError) as exc_info:
        _call_structured_llm(
            generator=generator,
            prompt='Please output JSON',
            stage_name='failing_stage',
            validator=lambda d: d['key'],
            max_retries=1,
        )
    assert 'failing_stage' in str(exc_info.value)


def test_router_single_and_multi_hop() -> None:
    gen1 = SequentialGenerator(['{"route": "single_hop", "query": "what is RAG", "reason": "simple", "requirements": [{"requirement_id": "R1", "description": "define RAG"}]}'])
    dec1, _ = route_query_llm(gen1, 'what is RAG')
    assert dec1.route == 'single_hop'
    assert dec1.query == 'what is RAG'

    gen2 = SequentialGenerator(['{"route": "multi_hop", "query": "who founded Acme", "reason": "two entities", "requirements": [{"requirement_id": "R1", "description": "identify founder"}, {"requirement_id": "R2", "description": "identify founder school"}]}'])
    dec2, _ = route_query_llm(gen2, 'Where did Acme founder go to school')
    assert dec2.route == 'multi_hop'
    assert dec2.query == 'who founded Acme'
    assert [item.requirement_id for item in dec2.requirements] == ['R1', 'R2']


def test_router_allows_more_requirements_than_hop_budget() -> None:
    route_payload = {
        'route': 'multi_hop',
        'query': 'first clue',
        'reason': 'multi',
        'requirements': [
            {'requirement_id': f'R{i}', 'description': f'condition {i}'} for i in range(1, 5)
        ],
    }
    decision, calls = route_query_llm(SequentialGenerator([json.dumps(route_payload)]), 'complex question')
    assert calls == 1
    assert len(decision.requirements) == 4


def test_evidence_grader_verdicts() -> None:
    requirements = [EvidenceRequirement('R1', 'identify founder')]
    gen_comp = SequentialGenerator(['{"verdict": "complete", "requirement_assessments": [{"requirement_id": "R1", "status": "supported", "evidence_chunk_ids": ["c1"]}], "reason": "done"}'])
    d_comp, _ = grade_evidence_llm(gen_comp, 'who founded A', 'A founder', requirements, [_hit('c1', 'Bob founded A', 1)])
    assert d_comp.verdict == 'complete'
    assert 'confirmed_facts' not in d_comp.to_dict()

    gen_cont = SequentialGenerator(['{"verdict": "continue", "requirement_assessments": [{"requirement_id": "R1", "status": "missing", "evidence_chunk_ids": []}], "next_requirement_id": "R1", "next_query": "Bob alma mater", "reason": "need university"}'])
    d_cont, _ = grade_evidence_llm(gen_cont, 'Bob school', 'A founder', requirements, [_hit('c1', 'Bob founded A', 1)])
    assert d_cont.verdict == 'continue'
    assert d_cont.next_query == 'Bob alma mater'

    gen_ins = SequentialGenerator(['{"verdict": "insufficient", "requirement_assessments": [{"requirement_id": "R1", "status": "missing", "evidence_chunk_ids": []}], "failure_reason": "no mention of Bob", "reason": "missing"}'])
    d_ins, _ = grade_evidence_llm(gen_ins, 'Bob school', 'Bob info', requirements, [_hit('c1', 'random info', 1)])
    assert d_ins.verdict == 'insufficient'
    assert d_ins.failure_reason == 'no mention of Bob'


def test_complete_gate_blocks_premature_completion_and_grounds_chunk_ids() -> None:
    requirements = [EvidenceRequirement('R1', 'first condition'), EvidenceRequirement('R2', 'second condition')]
    decision = EvidenceDecision(
        verdict='complete',
        model_verdict='complete',
        reason='answer entity is known',
        requirement_assessments=[
            RequirementAssessment('R1', 'supported', ['c1']),
            RequirementAssessment('R2', 'supported', ['hallucinated-chunk']),
        ],
    )
    gated = apply_complete_gate(decision, requirements, [_hit('c1', 'first condition', 1)])
    assert gated.model_verdict == 'complete'
    assert gated.verdict == 'continue'
    assert gated.next_requirement_id == 'R2'
    assert gated.next_query == 'second condition'
    assert gated.requirement_assessments[1].status == 'missing'
    assert gated.requirement_assessments[1].evidence_chunk_ids == []


def test_query_corrector() -> None:
    gen_corr = SequentialGenerator(['{"corrected_query": "Bob biography university degree", "reason": "broader search"}'])
    q_corr, _ = correct_query_llm(
        gen_corr,
        'Bob school',
        'Bob info',
        'no mention of school',
        [EvidenceRequirement('R1', 'identify Bob university')],
    )
    assert q_corr == 'Bob biography university degree'


class MockEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 16), dtype=np.float32) / np.sqrt(16)


class MockReranker:
    def rerank(self, query: str, hits: list[SearchHit], top_n: int) -> list[SearchHit]:
        return hits[:top_n]


def _create_mock_index(tmp_path: Path) -> Path:
    config_file = tmp_path / 'config' / 'default.yaml'
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text('\n'.join([
        'paths:',
        '  corpus_dir: data/raw',
        '  index_dir: data/index',
        '  runs_dir: runs',
        'models:',
        '  api_base: https://example.com',
        '  embedding_model: test-embed',
        '  embedding_dimensions: 16',
        '  llm_model: test-llm',
        '  timeout_seconds: 10',
        '  max_retries: 2',
        '  retry_delay_seconds: 1',
        'ingestion:',
        '  allowed_extensions: [".md"]',
        '  chunk_size: 200',
        '  chunk_overlap: 20',
        '  embedding_batch_size: 4',
        '  fail_on_error: true',
        'retrieval:',
        '  top_k: 4',
        '  min_score: 0.0',
        '  max_context_characters: 1000',
        'reranker:',
        '  enabled: true',
        '  model: test-rerank',
        '  top_n: 4',
        'generation:',
        '  temperature: 0.0',
        '  max_tokens: 100',
        'runtime:',
        '  save_runs: false',
        '  default_profile: hybrid-rerank',
        'agentic:',
        '  base_profile: hybrid-rerank',
        '  max_hops: 3',
        '  max_corrections_per_hop: 1',
        '  structured_output_retries: 1',
        '  recursion_limit: 20',
    ]), encoding='utf-8')

    chunks = [
        _chunk('c1', 'Bob founded Acme Corporation in 2010.', 'acme.md'),
        _chunk('c2', 'Bob graduated from Stanford University with a degree in CS.', 'bob.md'),
        _chunk('c3', 'Stanford is located in California.', 'stanford.md'),
    ]
    vectors = np.ones((len(chunks), 16), dtype=np.float32) / np.sqrt(16)
    meta = IndexMetadata('test-embed', 16, 'hash123', len(chunks), 200, 20)
    index_dir = tmp_path / 'data' / 'index'
    FaissStore.build(vectors, chunks, meta).save(index_dir)
    BM25Store.build(chunks, 'hash123').save(index_dir)
    return config_file


def test_agentic_workflow_multi_hop_chain(tmp_path: Path) -> None:
    config_file = _create_mock_index(tmp_path)
    config = load_config(config_file)

    # 1. Router -> multi_hop ('who founded Acme')
    # 2. Grader hop 1 -> continue ('Bob founded Acme', next_query: 'Bob university')
    # 3. Grader hop 2 -> complete ('Bob graduated from Stanford')
    responses = [
        json.dumps({'route': 'multi_hop', 'query': 'Acme founder', 'reason': 'multi', 'requirements': [{'requirement_id': 'R1', 'description': 'identify Acme founder'}, {'requirement_id': 'R2', 'description': 'identify founder university'}]}),
        json.dumps({'verdict': 'continue', 'requirement_assessments': [{'requirement_id': 'R1', 'status': 'supported', 'evidence_chunk_ids': ['c1']}, {'requirement_id': 'R2', 'status': 'missing', 'evidence_chunk_ids': []}], 'next_requirement_id': 'R2', 'next_query': 'Bob university', 'reason': 'found founder'}),
        json.dumps({'verdict': 'complete', 'requirement_assessments': [{'requirement_id': 'R1', 'status': 'supported', 'evidence_chunk_ids': ['c1']}, {'requirement_id': 'R2', 'status': 'supported', 'evidence_chunk_ids': ['c2']}], 'reason': 'found school'}),
    ]
    generator = SequentialGenerator(responses)
    embedder = MockEmbedder()
    reranker = MockReranker()

    result = run_agentic_retrieval(
        config=config,
        embedder=embedder,
        generator=generator,
        question='Where did Acme founder study?',
        reranker=reranker,
    )

    assert result.termination_reason == 'complete'
    assert result.total_hops == 2
    assert result.correction_count == 0
    assert len(result.traces) == 2
    assert len(result.final_hits) > 0


def test_agentic_workflow_complete_gate_forces_missing_requirement_hop(tmp_path: Path) -> None:
    config_file = _create_mock_index(tmp_path)
    config = load_config(config_file)
    responses = [
        json.dumps({'route': 'multi_hop', 'query': 'Acme founder', 'reason': 'two conditions', 'requirements': [{'requirement_id': 'R1', 'description': 'identify Acme founder'}, {'requirement_id': 'R2', 'description': 'identify founder university'}]}),
        json.dumps({'verdict': 'complete', 'requirement_assessments': [{'requirement_id': 'R1', 'status': 'supported', 'evidence_chunk_ids': ['c1']}, {'requirement_id': 'R2', 'status': 'missing', 'evidence_chunk_ids': []}], 'reason': 'answer entity is already known'}),
        json.dumps({'verdict': 'complete', 'requirement_assessments': [{'requirement_id': 'R1', 'status': 'supported', 'evidence_chunk_ids': ['c1']}, {'requirement_id': 'R2', 'status': 'supported', 'evidence_chunk_ids': ['c2']}], 'reason': 'all conditions covered'}),
    ]

    result = run_agentic_retrieval(
        config=config,
        embedder=MockEmbedder(),
        generator=SequentialGenerator(responses),
        question='Where did the founder of Acme study?',
        reranker=MockReranker(),
    )

    assert result.termination_reason == 'complete'
    assert result.total_hops == 2
    assert result.traces[0].decision is not None
    assert result.traces[0].decision.model_verdict == 'complete'
    assert result.traces[0].decision.verdict == 'continue'
    assert result.traces[0].decision.next_requirement_id == 'R2'
    assert result.traces[1].decision is not None
    assert result.traces[1].decision.verdict == 'complete'


def test_agentic_workflow_correction_leading_to_recovery(tmp_path: Path) -> None:
    config_file = _create_mock_index(tmp_path)
    config = load_config(config_file)

    # 1. Router -> single_hop
    # 2. Grader hop 1 (first try) -> insufficient
    # 3. Corrector -> new query 'Acme Corporation Bob'
    # 4. Grader hop 1 (after correction) -> complete
    responses = [
        json.dumps({'route': 'single_hop', 'query': 'Acme', 'reason': 'single', 'requirements': [{'requirement_id': 'R1', 'description': 'identify Acme founder'}]}),
        json.dumps({'verdict': 'insufficient', 'requirement_assessments': [{'requirement_id': 'R1', 'status': 'missing', 'evidence_chunk_ids': []}], 'failure_reason': 'not specific', 'reason': 'bad hits'}),
        json.dumps({'corrected_query': 'Acme Corporation Bob', 'reason': 'add name'}),
        json.dumps({'verdict': 'complete', 'requirement_assessments': [{'requirement_id': 'R1', 'status': 'supported', 'evidence_chunk_ids': ['c1']}], 'reason': 'done'}),
    ]
    generator = SequentialGenerator(responses)
    embedder = MockEmbedder()
    reranker = MockReranker()

    result = run_agentic_retrieval(
        config=config,
        embedder=embedder,
        generator=generator,
        question='Acme founder',
        reranker=reranker,
    )

    assert result.termination_reason == 'complete'
    assert result.total_hops == 1
    assert result.correction_count == 1
    assert len(result.traces) == 2


def test_agentic_workflow_max_hops_exhausted(tmp_path: Path) -> None:
    config_file = _create_mock_index(tmp_path)
    config = load_config(config_file)

    # Grader keeps returning continue
    responses = [
        json.dumps({'route': 'multi_hop', 'query': 'q1', 'reason': 'm', 'requirements': [{'requirement_id': 'R1', 'description': 'first'}, {'requirement_id': 'R2', 'description': 'second'}]}),
        json.dumps({'verdict': 'continue', 'requirement_assessments': [{'requirement_id': 'R1', 'status': 'missing', 'evidence_chunk_ids': []}, {'requirement_id': 'R2', 'status': 'missing', 'evidence_chunk_ids': []}], 'next_requirement_id': 'R1', 'next_query': 'q2', 'reason': 'step1'}),
        json.dumps({'verdict': 'continue', 'requirement_assessments': [{'requirement_id': 'R1', 'status': 'missing', 'evidence_chunk_ids': []}, {'requirement_id': 'R2', 'status': 'missing', 'evidence_chunk_ids': []}], 'next_requirement_id': 'R1', 'next_query': 'q3', 'reason': 'step2'}),
        json.dumps({'verdict': 'continue', 'requirement_assessments': [{'requirement_id': 'R1', 'status': 'missing', 'evidence_chunk_ids': []}, {'requirement_id': 'R2', 'status': 'missing', 'evidence_chunk_ids': []}], 'next_requirement_id': 'R1', 'next_query': 'q4', 'reason': 'step3'}),
    ]
    generator = SequentialGenerator(responses)
    embedder = MockEmbedder()
    reranker = MockReranker()

    result = run_agentic_retrieval(
        config=config,
        embedder=embedder,
        generator=generator,
        question='chain question',
        reranker=reranker,
    )

    assert result.termination_reason == 'max_hops_reached'
    assert result.total_hops == 3


def test_multihop_stratified_split(tmp_path: Path) -> None:
    dataset_dir = tmp_path / 'dataset'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'benchmark': 'MultiHop-RAG',
        'queries': [
            {'id': f'inf-{i:03d}', 'query': f'inf q {i}', 'answer': 'ans', 'question_type': 'inference_query', 'evidence_list': []} for i in range(70)
        ] + [
            {'id': f'comp-{i:03d}', 'query': f'comp q {i}', 'answer': 'ans', 'question_type': 'comparison_query', 'evidence_list': []} for i in range(70)
        ] + [
            {'id': f'temp-{i:03d}', 'query': f'temp q {i}', 'answer': 'ans', 'question_type': 'temporal_query', 'evidence_list': []} for i in range(70)
        ] + [
            {'id': 'null-001', 'query': 'null', 'answer': 'ans', 'question_type': 'null_query', 'evidence_list': []}
        ]
    }
    (dataset_dir / 'multihoprag-manifest.json').write_text(json.dumps(manifest), encoding='utf-8')

    splits = get_or_create_multihop_splits(dataset_dir, seed=42, dev_count_per_type=20, test_count_per_type=40)
    assert len(splits['dev']) == 60
    assert len(splits['test']) == 120

    dev_types = [q['question_type'] for q in splits['dev']]
    assert dev_types.count('inference_query') == 20
    assert dev_types.count('comparison_query') == 20
    assert dev_types.count('temporal_query') == 20

    test_types = [q['question_type'] for q in splits['test']]
    assert test_types.count('inference_query') == 40
    assert test_types.count('comparison_query') == 40
    assert test_types.count('temporal_query') == 40

    # Non-overlapping
    dev_ids = {q['id'] for q in splits['dev']}
    test_ids = {q['id'] for q in splits['test']}
    assert dev_ids.isdisjoint(test_ids)


def test_paired_deltas_calculation() -> None:
    baseline_records = [
        {'id': 'q1', 'status': 'ok', 'evidence_coverage_at_10': 0.5, 'complete_evidence_at_10': False, 'official_mrr_at_10': 0.5, 'evidence_coverage_at_4': 0.5, 'complete_evidence_at_4': False},
        {'id': 'q2', 'status': 'ok', 'evidence_coverage_at_10': 1.0, 'complete_evidence_at_10': True, 'official_mrr_at_10': 1.0, 'evidence_coverage_at_4': 1.0, 'complete_evidence_at_4': True},
    ]
    agentic_records = [
        {'id': 'q1', 'status': 'ok', 'evidence_coverage_at_10': 1.0, 'complete_evidence_at_10': True, 'official_mrr_at_10': 1.0, 'evidence_coverage_at_4': 1.0, 'complete_evidence_at_4': True},
        {'id': 'q2', 'status': 'ok', 'evidence_coverage_at_10': 1.0, 'complete_evidence_at_10': True, 'official_mrr_at_10': 1.0, 'evidence_coverage_at_4': 1.0, 'complete_evidence_at_4': True},
    ]
    deltas = _calc_paired_deltas(baseline_records, agentic_records)
    assert deltas['common_questions'] == 2
    assert deltas['complete_evidence_at_10']['mean_delta'] == 0.5
    assert deltas['evidence_coverage_at_10']['mean_delta'] == 0.25
    assert deltas['complete_evidence_at_10']['improved_count'] == 1
    assert deltas['complete_evidence_at_10']['tied_count'] == 1
    assert deltas['complete_evidence_at_10']['degraded_count'] == 0


def test_agentic_workflow_router_failure_handled(tmp_path: Path) -> None:
    config_file = _create_mock_index(tmp_path)
    config = load_config(config_file)
    # Router fails all retries
    generator = SequentialGenerator(['bad 1', 'bad 2'])
    result = run_agentic_retrieval(
        config=config,
        embedder=MockEmbedder(),
        generator=generator,
        question='broken question',
        reranker=MockReranker(),
    )
    assert result.termination_reason == 'router_failed'


def test_agentic_workflow_grader_failure_handled(tmp_path: Path) -> None:
    config_file = _create_mock_index(tmp_path)
    config = load_config(config_file)
    # Router ok, but grader fails
    generator = SequentialGenerator([
        json.dumps({'route': 'single_hop', 'query': 'q', 'reason': 'ok', 'requirements': [{'requirement_id': 'R1', 'description': 'answer question'}]}),
        'bad grader 1',
        'bad grader 2',
    ])
    result = run_agentic_retrieval(
        config=config,
        embedder=MockEmbedder(),
        generator=generator,
        question='question',
        reranker=MockReranker(),
    )
    assert result.termination_reason == 'grader_failed'


def test_agentic_workflow_max_corrections_exhausted(tmp_path: Path) -> None:
    config_file = _create_mock_index(tmp_path)
    config = load_config(config_file)
    # Router ok, Grader insufficient -> Corrector ok -> Grader insufficient again (max_corrections=1 reached)
    generator = SequentialGenerator([
        json.dumps({'route': 'single_hop', 'query': 'q', 'reason': 'ok', 'requirements': [{'requirement_id': 'R1', 'description': 'answer question'}]}),
        json.dumps({'verdict': 'insufficient', 'requirement_assessments': [{'requirement_id': 'R1', 'status': 'missing', 'evidence_chunk_ids': []}], 'failure_reason': 'missing', 'reason': 'r'}),
        json.dumps({'corrected_query': 'q2', 'reason': 'better'}),
        json.dumps({'verdict': 'insufficient', 'requirement_assessments': [{'requirement_id': 'R1', 'status': 'missing', 'evidence_chunk_ids': []}], 'failure_reason': 'still missing', 'reason': 'r2'}),
    ])
    result = run_agentic_retrieval(
        config=config,
        embedder=MockEmbedder(),
        generator=generator,
        question='question',
        reranker=MockReranker(),
    )
    assert result.termination_reason == 'insufficient_evidence'
    assert result.correction_count == 1


def test_evaluate_agentic_multihop_validation_and_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = _create_mock_index(tmp_path)
    config = load_config(config_file)
    # Invalid split
    with pytest.raises(ValueError, match='split 只能是'):
        from base_rag.agentic_eval import evaluate_agentic_multihop
        evaluate_agentic_multihop(config, MockEmbedder(), SequentialGenerator([]), split='invalid')
    # Invalid system
    with pytest.raises(ValueError, match='system 只能是'):
        from base_rag.agentic_eval import evaluate_agentic_multihop
        evaluate_agentic_multihop(config, MockEmbedder(), SequentialGenerator([]), split='dev', system='invalid')

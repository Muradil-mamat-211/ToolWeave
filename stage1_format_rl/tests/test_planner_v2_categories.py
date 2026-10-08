from __future__ import annotations

import asyncio
import copy
import json
import os
from pathlib import Path
import sys
import subprocess

import pytest
from jsonschema import ValidationError

from env_tuning.rods_data_generation_v1.config import LLMConfig
from env_tuning.rods_data_generation_v1.contracts import planner_input_validator
from env_tuning.rods_data_generation_v1.daemon import GeneratorDaemon
from env_tuning.rods_data_generation_v1.codex_backend import CodexCLIBackend
from env_tuning.rods_data_generation_v1.environment_adapter import SynthesisEnvironmentAdapter
from env_tuning.rods_data_generation_v1.error_taxonomy import ErrorType
from env_tuning.rods_data_generation_v1.execution_orchestrator import StageFailure
from env_tuning.rods_data_generation_v1.llm_backend import BackendError, BackendQuotaExceeded, FakeLLMBackend
from env_tuning.rods_data_generation_v1.metrics import GeneratorMetrics
from env_tuning.rods_data_generation_v1.models import SeedRecord, FunctionCall, ErrorRecord
from env_tuning.rods_data_generation_v1.parsing import StructuredParseError
from env_tuning.rods_data_generation_v1.parameter_generator import ParameterGenerator
from env_tuning.rods_data_generation_v1.pipeline import RODSDataGenerationPipeline
from env_tuning.rods_data_generation_v1.planner import PlannerAgent
from env_tuning.rods_data_generation_v1.planner_contract import build_planner_input
from env_tuning.rods_data_generation_v1.queue import LockedJsonlQueue, atomic_write_json
from env_tuning.rods_data_generation_v1.quality_judge import conversation_summary
from env_tuning.rods_data_generation_v1.query_verifier import QueryVerifier
from env_tuning.rods_data_generation_v1.query_generator import QueryGenerator
from env_tuning.rods_data_generation_v1.quality_judge import QualityJudgeAgent, deterministic_leakage_reason
from env_tuning.rods_data_generation_v1.adversarial.final_trace import planner_scaffold_alignment
from env_tuning.rods_data_generation_v1.validation.semantic_grounding import semantic_context_for_verifier
from env_tuning.rods_data_generation_v1.validation.semantic_grounding import semantic_grounding_gate
from env_tuning.rods_matchtir_v1.provenance import extract_source_tool_updates
from rods_data_generation_v1_fixtures import VERIFY_ACCEPT, make_catalog, make_seed, make_config, success_script
from test_rods_generator_semantic_hardening import _draft, _record, _turn


@pytest.mark.parametrize('kind', [
    'multi_turn_base', 'multi_turn_miss_func', 'multi_turn_miss_param', 'multi_turn_long_context',
])
def test_final_verifier_receives_category_and_actual_tool_recovery_timeline(kind):
    catalog = make_catalog()
    normal = _turn(0, 'GorillaFileSystem', 'List the directory.', [
        _record('ls', {}, {'files': ['deploy.py']}, class_name='GorillaFileSystem', turn_id=0, call_id=0)
    ])
    recovery = _turn(1, 'GorillaFileSystem', 'Read deploy.py.', [
        _record('cat', {'file_name': 'deploy.py'}, {'content': 'def deploy(): pass'},
                class_name='GorillaFileSystem', turn_id=1, call_id=0)
    ])
    draft = _draft([normal, recovery], data_type=kind)
    draft.initial_tools = [catalog.get('ls').schema]
    recovery.recovery_tools = [catalog.get('cat').schema]
    backend = FakeLLMBackend({'final_query_verifier': ['<reason>inspect evidence</reason><verdict>accept</verdict>']})
    asyncio.run(QueryVerifier(backend, GeneratorMetrics()).verify_final_conversation(draft))
    prompt = backend.calls[0]['messages'][0]['content']
    payload, _ = json.JSONDecoder().raw_decode(prompt[prompt.index('{'):])
    assert payload['data_type'] == kind
    assert {schema['name'] for schema in payload['initial_tool_definitions']} == {'ls'}
    assert payload['turns'][0]['recovery_tool_definitions'] == []
    assert payload['turns'][1]['recovery_tool_definitions'] == [catalog.get('cat').schema]
    assert 'do not supply a missing actor-visible parameter' in payload['actor_visibility_contract']['hidden_evidence']


@pytest.mark.parametrize('kind,max_turns', [
    ('multi_turn_base', 5), ('multi_turn_miss_func', 4),
    ('multi_turn_miss_param', 4), ('multi_turn_long_context', 5),
])
def test_planner_whitelists_task_fields_and_routes_only_current_category(kind, max_turns):
    raw = make_seed(kind)
    raw['generation_metadata']['old_rollouts'] = ['DO_NOT_SEND_POLICY_HISTORY']
    raw['generation_metadata']['structural_profile'] = {'secret': 'DO_NOT_SEND_TOPOLOGY'}
    seed = SeedRecord.from_mapping(raw)
    agent = PlannerAgent(FakeLLMBackend({}), make_catalog(), GeneratorMetrics())
    prompt, names = agent._render(seed, failure_history=[], blocked_functions=set(), current_config=seed.initial_config)
    payload = agent.rendered_inputs[-1]
    assert payload['data_type'] == kind
    assert payload['generation_contract']['maximum_planned_turns'] == max_turns
    assert payload['generation_contract']['argument_limits'] == {'max_string_characters': 200, 'max_collection_items': 5}
    assert 'DO_NOT_SEND_POLICY_HISTORY' not in prompt and 'DO_NOT_SEND_TOPOLOGY' not in prompt
    assert 'PROJECT_STRUCTURAL_GUIDANCE' not in prompt
    assert names == [s['name'] for s in seed.available_functions]
    assert payload['original_task']['initial_tool_definitions'] == seed.available_functions
    assert set(payload['original_task']) == {'user_queries', 'reference_ground_truth', 'initial_tool_definitions', 'tool_updates'}
    assert ('Fix the future user\'s concrete choice' in prompt) is (kind == 'multi_turn_miss_param')


def test_source_missing_tool_is_restored_once_and_schema_checked():
    catalog = make_catalog()
    raw = make_seed('multi_turn_miss_func')
    schema = copy.deepcopy(catalog.get('multiply').schema)
    raw['available_functions'] = [s for s in raw['available_functions'] if s['name'] != 'multiply']
    original_update = json.dumps([schema]) + '\nI have updated some more functions you can choose from. What about now?'
    updates = extract_source_tool_updates({'processed_question': [original_update]})
    raw['generation_metadata']['planner_source'] = {'tool_updates': updates}
    seed = SeedRecord.from_mapping(raw)
    payload = build_planner_input(seed, catalog, current_config={})
    assert 'multiply' not in {s['name'] for s in payload['original_task']['initial_tool_definitions']}
    assert payload['original_task']['tool_updates'] == [{'turn_id': 1, 'tools': [schema]}]
    raw['generation_metadata']['planner_source']['tool_updates'][0]['tools'][0]['description'] = 'fabricated definition'
    with pytest.raises(ValueError, match='differs from VM catalog'):
        build_planner_input(SeedRecord.from_mapping(raw), catalog, current_config={})


def test_missing_categories_reserve_recovery_turn_in_real_parser():
    seed = SeedRecord.from_mapping(make_seed('multi_turn_miss_param'))
    five_turns = '<reason>r</reason><narrative>n</narrative>' + '<turn>MathAPI: add</turn>' * 5
    agent = PlannerAgent(FakeLLMBackend({'planner': [five_turns]}), make_catalog(), GeneratorMetrics(), max_parse_retries=1)
    with pytest.raises(StructuredParseError, match='2-4 turns'):
        asyncio.run(agent.plan(seed, failure_history=[], blocked_functions=set(), current_config={}))


@pytest.mark.parametrize('mutation', ['rollouts', 'wrong_recovery_budget', 'wrong_long_context'])
def test_input_schema_rejects_unapproved_content_and_category_inconsistency(mutation):
    seed = SeedRecord.from_mapping(make_seed('multi_turn_miss_param'))
    payload = build_planner_input(seed, make_catalog(), current_config={})
    if mutation == 'rollouts':
        payload['original_task']['rollouts'] = []
    elif mutation == 'wrong_recovery_budget':
        payload['generation_contract']['maximum_planned_turns'] = 5
    else:
        payload['initial_environment']['long_context'] = True
    with pytest.raises(ValidationError):
        planner_input_validator().validate(payload)


def test_generator_import_does_not_require_training_gpu_dependencies():
    source = Path(__file__).resolve().parents[2] / 'code/AWorld-RL-stage1-worktree/EnvTuning'
    code = '''import importlib.abc, sys
class BlockTraining(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch', 'scipy'}:
   raise AssertionError('Generator imported a Training dependency: ' + fullname)
sys.meta_path.insert(0, BlockTraining())
from env_tuning.rods_data_generation_v1 import RODSDataGenerationPipeline
from env_tuning.rods_matchtir_v1.lifecycle import validate_candidate_record
from env_tuning.rods_matchtir_v1.provenance import extract_source_tool_updates
'''
    result = subprocess.run([sys.executable, '-c', code], env={**os.environ, 'PYTHONPATH': str(source)},
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_long_context_label_without_large_observations_is_rejected(tmp_path):
    raw = make_seed('multi_turn_base')
    raw['data_type'] = 'multi_turn_long_context'
    backend = FakeLLMBackend(success_script())
    pipeline = RODSDataGenerationPipeline(config=make_config(tmp_path=tmp_path),
        backend=backend, catalog=make_catalog(), environment_factory=SynthesisEnvironmentAdapter())
    result = asyncio.run(pipeline.generate(raw))
    assert result.status == 'DROPPED'
    assert 'large prior actor-visible observation' in result.reason
    assert not any(call['role'] == 'quality_judge' for call in backend.calls)


def test_parameter_agent_rejects_overlong_message_before_vm_execution():
    catalog = make_catalog()
    arguments = {'receiver_id': 'USR003', 'message': 'x' * 202}
    backend = FakeLLMBackend({'parameter_generator': [
        '<reason>send a review</reason><arguments>' + json.dumps(arguments) + '</arguments>'
    ]})
    agent = ParameterGenerator(backend, catalog, GeneratorMetrics())
    with pytest.raises(StructuredParseError, match='string exceeds 200 characters'):
        asyncio.run(agent.generate(spec=catalog.get('send_message'), environment_state={},
                                  execution_history=[], narrative='Send a compact review.', turn_id=0))
    assert '"max_string_characters": 200' in backend.calls[0]['messages'][0]['content']


def test_quality_context_preserves_actual_delete_effect_despite_return_wording():
    config = {'MessageAPI': {'current_user': 'USR002', 'message_count': 1,
                             'inbox': [{'USR003': 'older message'}]}}
    session = SynthesisEnvironmentAdapter().create(initial_config=config,
        involved_classes=['MessageAPI'], seed_id='judge-evidence', long_context=False,
        purpose='judge_state_evidence')
    try:
        calls = [FunctionCall('send_message', {'receiver_id': 'USR003', 'message': 'task message'}, 'MessageAPI'),
                 FunctionCall('delete_message', {'receiver_id': 'USR003'}, 'MessageAPI')]
        results = [session.execute(call) for call in calls]
        assert all(result.success for result in results)
        turns = [_turn(index, 'MessageAPI', query, [_record(call.name, call.arguments, result.result,
            class_name='MessageAPI', turn_id=index, call_id=0,
            pre_state=result.pre_state, post_state=result.post_state)])
            for index, (call, result, query) in enumerate(zip(calls, results,
                ['Send Catherine a task message.', 'Delete only the latest task message.']))]
        draft = _draft(turns)
        catalog = make_catalog()
        draft.initial_tools = [catalog.get(call.name).schema for call in calls]
        summary = conversation_summary(draft)
        evidence = summary['verification_evidence']
        deleted = evidence['execution_trace'][1]
        assert 'first message' in deleted['result']['message']
        assert deleted['pre_state']['MessageAPI']['inbox'] == [
            {'USR003': 'older message'}, {'USR003': 'task message'}]
        assert deleted['post_state']['MessageAPI']['inbox'] == [{'USR003': 'older message'}]
        assert {tool['name'] for tool in evidence['used_tool_definitions']} == {'send_message', 'delete_message'}
        assert 'not extra actor-visible history' in evidence['visibility_note']
        backend = FakeLLMBackend({'query_verifier': ['<reason>actual deletion effect</reason><verdict>accept</verdict>']})
        asyncio.run(QueryVerifier(backend, GeneratorMetrics(), catalog=catalog).verify(
            query=turns[1].query, turn_records=turns[1].execution_records, execution_context=session.snapshot()))
        context_text = backend.calls[0]['messages'][0]['content'].split('Relevant environment/dependency context:\n', 1)[1]
        context, _ = json.JSONDecoder().raw_decode(context_text)
        per_turn = context['verification_evidence']
        assert per_turn['execution_trace'][0]['pre_state']['MessageAPI']['inbox'] == deleted['pre_state']['MessageAPI']['inbox']
        assert per_turn['execution_trace'][0]['post_state']['MessageAPI']['inbox'] == deleted['post_state']['MessageAPI']['inbox']
        assert [tool['name'] for tool in per_turn['used_tool_definitions']] == ['delete_message']
    finally:
        session.close()


@pytest.mark.parametrize('file_name,pattern,query,prior_file,passed', [
    ('demo.py', 'def ', 'Show the line declaring its function in demo.py.', None, True),
    ('demo.py', 'def', 'Show the function declaration in demo.py.', None, True),
    ('demo.py', 'def deploy', 'Show the function declaration in demo.py.', 'demo.py', True),
    ('demo.py', 'def deploy', 'Show the line defining its deployment function in demo.py.', 'demo.py', True),
    ('demo.py', 'def deploy', 'Show the function declaration in demo.py.', None, False),
    ('demo.py', 'def missing', 'Show the function declaration in demo.py.', 'demo.py', False),
    ('demo.py', 'def deploy', 'Show the function declaration in demo.py.', 'other.py', False),
    ('demo.txt', 'def ', 'Show the function declaration in demo.txt.', None, False),
    ('demo.py', 'def ', 'Find the update-related lines in demo.py.', None, False),
    ('demo.py', 'arbitrary', 'Show the function declaration in demo.py.', 'demo.py', False),
])
def test_python_declaration_grounding_is_scoped_and_named_patterns_need_prior_evidence(file_name, pattern, query, prior_file, passed):
    turns = []
    if prior_file:
        turns.append(_turn(0, 'GorillaFileSystem', f'Read {prior_file}.', [
            _record('cat', {'file_name': prior_file}, {'file_content': 'def deploy(): pass'},
                    class_name='GorillaFileSystem', turn_id=0, call_id=0)]))
    turn_id = len(turns)
    turns.append(_turn(turn_id, 'GorillaFileSystem', query, [
        _record('grep', {'file_name': file_name, 'pattern': pattern}, {'matching_lines': ['def deploy(): pass']},
                class_name='GorillaFileSystem', turn_id=turn_id, call_id=0)]))
    result = semantic_grounding_gate(_draft(turns), catalog=make_catalog())
    assert result.passed is passed, result.detail


@pytest.mark.parametrize('kind,includes_history', [('multi_turn_base', True), ('multi_turn_long_context', True)])
def test_long_context_query_uses_only_actual_prior_observations(kind, includes_history):
    prior = _record('cat', {'file_name': 'demo.py'}, {'file_content': 'ACTUAL_LARGE_OBSERVATION ' + 'x' * 2200},
        class_name='GorillaFileSystem', turn_id=0, call_id=0,
        pre_state={'secret': 'DO_NOT_SEND_INTERNAL_SNAPSHOT'})
    current = _record('send_message', {'receiver_id': 'USR003', 'message': 'summary'}, {'sent_status': True},
        class_name='MessageAPI', turn_id=1, call_id=0)
    backend = FakeLLMBackend({'query_generator': ['<reason>use prior report</reason><query>Send Catherine a summary of that report.</query>']})
    asyncio.run(QueryGenerator(backend, make_catalog(), GeneratorMetrics()).generate(
        class_name='MessageAPI', narrative='Read a report and send its relevant facts.', turn_records=[current],
        prior_queries=['Read demo.py.'], prior_records=[prior], data_type=kind))
    prompt = backend.calls[0]['messages'][0]['content']
    assert ('ACTUAL_LARGE_OBSERVATION' in prompt) is includes_history
    assert 'DO_NOT_SEND_INTERNAL_SNAPSHOT' not in prompt
    if kind == 'multi_turn_long_context':
        assert 'Do not paste the entire resolved message' in prompt


def test_judge_distinguishes_subject_code_from_tools_and_audits_real_return_mismatch():
    catalog = make_catalog()
    session = SynthesisEnvironmentAdapter().create(initial_config={'MessageAPI': {'current_user': 'USR002'}},
        involved_classes=['MessageAPI'], seed_id='return-contract', long_context=False, purpose='contract_audit')
    try:
        call = FunctionCall('send_message', {'receiver_id': 'USR003', 'message': 'Function deploy() has one line.'}, 'MessageAPI')
        result = session.execute(call)
        assert result.success
        turn = _turn(0, 'MessageAPI', 'Send Catherine this text: Function deploy() has one line.', [
            _record(call.name, call.arguments, result.result, class_name='MessageAPI', turn_id=0, call_id=0,
                    pre_state=result.pre_state, post_state=result.post_state)])
        draft = _draft([turn]); draft.initial_tools = [catalog.get('send_message').schema]
        assert deterministic_leakage_reason(draft) is None
        diagnostics = conversation_summary(draft)['verification_evidence']['response_contract_diagnostics']
        assert any(item['path'] == 'result.message_id' and item['declared_type'] == 'integer' and item['actual_type'] == 'dict' for item in diagnostics)
        backend = FakeLLMBackend({'quality_judge': ['<reason>verify actual effects</reason><decision>accept</decision><fail_reason></fail_reason>']})
        asyncio.run(QualityJudgeAgent(backend, GeneratorMetrics()).evaluate(draft))
        assert 'Names of functions in a user\'s source file' in backend.calls[0]['messages'][0]['content']
        assert 'response_contract_diagnostics' in backend.calls[0]['messages'][1]['content']
        assert 'Executable GT records tool calls' in backend.calls[0]['messages'][0]['content']
        turn.query = 'Call send_message to send this note.'
        assert deterministic_leakage_reason(draft) is not None
    finally:
        session.close()


def test_per_turn_and_final_verifiers_share_grounded_answer_presentation_rules():
    record = _record('list_users', {}, {'users': {'USR001': 'Alice', 'USR002': 'Bob', 'USR004': 'Daniel'}},
                     class_name='MessageAPI', turn_id=0, call_id=0)
    query = 'List workspace users and show the first and last distinct names after ascending lexical sorting.'
    draft = _draft([_turn(0, 'MessageAPI', query, [record])])
    backend = FakeLLMBackend({'query_verifier': [VERIFY_ACCEPT], 'final_query_verifier': [VERIFY_ACCEPT]})
    verifier = QueryVerifier(backend, GeneratorMetrics(), catalog=make_catalog())
    asyncio.run(verifier.verify(query=query, turn_records=[record], execution_context={}))
    asyncio.run(verifier.verify_final_conversation(draft))
    for call in backend.calls:
        prompt = call['messages'][0]['content']
        assert 'Executable GT records tool calls' in prompt
        assert 'Sending/writing/deleting/changing an object is a tool effect' in prompt
        assert 'fill a missing' in prompt and 'user preference' in prompt


def test_scaffold_alignment_uses_preserved_execution_origin_after_clarification():
    from dataclasses import replace
    first = _record('add', {'a': 2, 'b': 3}, 5, class_name='MathAPI', turn_id=1, call_id=0)
    first = replace(first, dependency_provenance={'synthesis_turn_id': 0})
    second = _record('multiply', {'a': 5, 'b': 2}, 10, class_name='MathAPI', turn_id=2, call_id=0)
    second = replace(second, dependency_provenance={'synthesis_turn_id': 1})
    missing = _turn(0, 'MathAPI', 'Add the numbers.', [])
    missing.is_intentional_missing = True
    draft = _draft([missing, _turn(1, 'MathAPI', 'Use 2 and 3.', [first]),
                    _turn(2, 'MathAPI', 'Multiply by 2.', [second])], data_type='multi_turn_miss_param')
    view = planner_scaffold_alignment(draft)
    assert view['intentional_missing_turn_ids'] == [0]
    assert [(row['scaffold_turn_id'], row['final_executable_turn_id']) for row in view['mapping']] == [(0, 1), (1, 2)]
    assert json.loads(semantic_context_for_verifier(draft))['planner_scaffold_alignment'] == view
    assert conversation_summary(draft)['planner_scaffold_alignment'] == view


def test_queue_and_checkpoint_preserve_bfcl_root_selection(tmp_path):
    config = {'GorillaFileSystem': {'root': {
        'project': {'type': 'directory', 'contents': {'deploy.py': {'type': 'file', 'content': 'real'}}},
        'backup_scripts': {'type': 'directory', 'contents': {}},
    }}}
    queue = LockedJsonlQueue(tmp_path/'seeds.jsonl')
    queue.append([{'initial_config': config}])
    atomic_write_json(tmp_path/'checkpoint.json', {'initial_config': config})
    restored = [queue.read()[0]['initial_config'], json.loads((tmp_path/'checkpoint.json').read_text())['initial_config']]
    for value in restored:
        assert list(value['GorillaFileSystem']['root']) == ['project', 'backup_scripts']
        factory = SynthesisEnvironmentAdapter()
        session = factory.create(initial_config=value, involved_classes=['GorillaFileSystem'], seed_id='root', long_context=False, purpose='serialization_check')
        try:
            result = session.execute(FunctionCall('cat', {'file_name': 'deploy.py'}, 'GorillaFileSystem'))
            assert result.success and result.result == {'file_content': 'real'}
        finally:
            session.close()


def _fake_codex(tmp_path):
    executable = tmp_path/'codex-fixture'
    executable.write_text(f'#!{sys.executable}\n' + '''import json, os, pathlib, sys, time
if sys.argv[1:3] == ['login', 'status']:
 print('Logged in using ChatGPT'); sys.exit(0)
data = sys.stdin.read()
pathlib.Path('env-check.json').write_text(json.dumps({k: k in os.environ for k in ['OPENAI_API_KEY','CODEX_API_KEY']}))
print(json.dumps({'type':'thread.started'}), flush=True)
print('fixture transport started', file=sys.stderr, flush=True)
if os.environ.get('CODEX_TEST_QUOTA'):
 print(json.dumps({'type':'error','message':'You have hit your usage limit. Try again later.'}), flush=True)
 print(json.dumps({'type':'turn.failed','error':{'message':'You have hit your usage limit.'}}), flush=True)
 sys.exit(1)
if os.environ.get('CODEX_TEST_DELAY'): time.sleep(15)
output = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])
output.write_text(json.dumps({'text':'<reason>verified inline role</reason>'}))
if os.environ.get('CODEX_TEST_TOOL_EVENT'):
 print(json.dumps({'type':'item.completed','item':{'type':'command_execution'}}))
print(json.dumps({'type':'turn.completed'}))
''')
    executable.chmod(0o755)
    return executable


def test_cli_transport_scrubs_api_keys_and_preserves_structured_text(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-value-never-use')
    monkeypatch.setenv('CODEX_API_KEY', 'test-value-never-use')
    backend = CodexCLIBackend(LLMConfig(backend='codex_cli', model='', codex_binary=str(_fake_codex(tmp_path)), codex_artifact_dir=str(tmp_path/'calls')))
    response = asyncio.run(backend.complete(role='planner', messages=[{'role':'user','content':'inline input'}]))
    assert response.text == '<reason>verified inline role</reason>'
    job = Path(response.raw_response['artifact_dir'])
    assert json.loads((job/'env-check.json').read_text()) == {'OPENAI_API_KEY':False,'CODEX_API_KEY':False}
    assert 'inline input' in (job/'prompt.stdin.txt').read_text()


def test_cli_quota_stops_queued_requests_without_spawning_more_processes(tmp_path, monkeypatch):
    monkeypatch.setenv('CODEX_TEST_QUOTA', '1')
    backend = CodexCLIBackend(LLMConfig(backend='codex_cli', model='', codex_binary=str(_fake_codex(tmp_path)),
        codex_artifact_dir=str(tmp_path/'calls'), concurrency=1))

    async def exhaust():
        results = await asyncio.gather(
            backend.complete(role='planner', messages=[]),
            backend.complete(role='parameter_generator', messages=[]), return_exceptions=True)
        assert all(isinstance(result, BackendQuotaExceeded) for result in results)
        with pytest.raises(BackendQuotaExceeded, match='account quota exhausted'):
            await backend.complete(role='query_generator', messages=[])

    asyncio.run(exhaust())
    assert len(list((tmp_path/'calls').glob('codex-*/env-check.json'))) == 1
    assert not backend._processes


@pytest.mark.parametrize('role,kind', [
    ('planner', 'multi_turn_base'), ('parameter_generator', 'multi_turn_base'),
    ('query_generator', 'multi_turn_base'), ('query_verifier', 'multi_turn_base'),
    ('coherence_rewrite', 'multi_turn_base'), ('final_query_verifier', 'multi_turn_base'),
    ('quality_judge', 'multi_turn_base'), ('missing_function', 'multi_turn_miss_func'),
    ('missing_parameter', 'multi_turn_miss_param'),
])
def test_quota_defers_seed_without_consuming_attempts_and_resume_completes(tmp_path, role, kind):
    config = make_config(tmp_path=tmp_path, dry_run=False)
    script = success_script(kind)
    script[role] = [BackendQuotaExceeded('account usage limit; retained audit evidence')]
    backend = FakeLLMBackend(script)
    pipeline = RODSDataGenerationPipeline(config=config, backend=backend, catalog=make_catalog())
    daemon = GeneratorDaemon(config=config, backend=backend, pipeline=pipeline)
    raw = make_seed(kind)
    daemon.seed_queue.append([raw])
    with pytest.raises(BackendQuotaExceeded):
        asyncio.run(daemon.run_once(allow_generation=True))
    state = daemon.tracker.snapshot()['seeds'][raw['sample_id']]
    assert state['status'] == 'PENDING'
    assert state['completed_failed_attempts'] == 0
    assert state['failures'] == [] and state['blocklist'] == []
    assert daemon.candidate_queue.read() == [] and daemon.terminal_results.read() == []
    assert sum(call['role'] == role for call in backend.calls) == 1
    assert daemon.expanded_results.read()[0]['status'] == 'DEFERRED_BACKEND_QUOTA'

    # A fresh backend after provider recovery uses the retained source seed and
    # checkpoint. This scripted fixture tests lifecycle, not model quality.
    recovered = FakeLLMBackend(success_script(kind))
    resumed_pipeline = RODSDataGenerationPipeline(config=config, backend=recovered, catalog=make_catalog())
    resumed = GeneratorDaemon(config=config, backend=recovered, pipeline=resumed_pipeline)
    asyncio.run(resumed.run_once(allow_generation=True))
    assert resumed.tracker.snapshot()['seeds'][raw['sample_id']]['status'] == 'SUCCEEDED'
    assert len(resumed.candidate_queue.read()) == 1
    assert len(resumed.terminal_results.read()) == 1


def test_config_patch_quota_preserves_the_prior_real_failure_only(tmp_path, monkeypatch):
    raw = make_seed()
    script = success_script()
    script['config_patch'] = [BackendQuotaExceeded('account usage limit')]
    backend = FakeLLMBackend(script)
    config = make_config(tmp_path=tmp_path, dry_run=False)
    pipeline = RODSDataGenerationPipeline(config=config, backend=backend, catalog=make_catalog())

    async def fail_execution(**kwargs):
        raise StageFailure(ErrorRecord(error_type=ErrorType.VM_EXEC_FAILED,
            seed_id=raw['sample_id'], attempt_id=1, turn_id=0, function_names=('add',),
            detail='controlled genuine VM failure before quota', patchable=True))

    monkeypatch.setattr(pipeline.execution, 'execute', fail_execution)
    daemon = GeneratorDaemon(config=config, backend=backend, pipeline=pipeline)
    daemon.seed_queue.append([raw])
    with pytest.raises(BackendQuotaExceeded):
        asyncio.run(daemon.run_once(allow_generation=True))
    state = daemon.tracker.snapshot()['seeds'][raw['sample_id']]
    assert state['status'] == 'PENDING'
    assert state['completed_failed_attempts'] == 1
    assert len(state['failures']) == 1 and state['failures'][0]['error_type'] == 'vm_exec_failed'
    assert state['blocklist'] == ['add']
    assert state['patches'] == []
    assert sum(call['role'] == 'planner' for call in backend.calls) == 1
    assert daemon.terminal_results.read() == []


def test_concurrent_quota_leaves_every_claimed_seed_pending(tmp_path):
    config = make_config(tmp_path=tmp_path, dry_run=False)
    backend = FakeLLMBackend({'planner': [BackendQuotaExceeded('account usage limit')] * 2})
    pipeline = RODSDataGenerationPipeline(config=config, backend=backend, catalog=make_catalog())
    daemon = GeneratorDaemon(config=config, backend=backend, pipeline=pipeline)
    seeds = [make_seed('multi_turn_base'), make_seed('multi_turn_miss_param')]
    daemon.seed_queue.append(seeds)
    with pytest.raises(BackendQuotaExceeded):
        asyncio.run(daemon.run_once(allow_generation=True))
    states = daemon.tracker.snapshot()['seeds']
    assert len(states) == 2
    assert all(state['status'] == 'PENDING' and state['completed_failed_attempts'] == 0 for state in states.values())
    assert daemon.candidate_queue.read() == [] and daemon.terminal_results.read() == []


@pytest.mark.parametrize('failure', ['CODEX_TEST_TOOL_EVENT', 'CODEX_TEST_DELAY'])
def test_cli_fails_closed_on_tool_use_or_timeout(tmp_path, monkeypatch, failure):
    monkeypatch.setenv(failure, '1')
    backend = CodexCLIBackend(LLMConfig(backend='codex_cli', model='', codex_binary=str(_fake_codex(tmp_path)), codex_artifact_dir=str(tmp_path/'calls'), timeout_seconds=2))
    with pytest.raises(BackendError):
        asyncio.run(backend.complete(role='planner', messages=[]))
    assert not backend._processes
    if failure == 'CODEX_TEST_DELAY':
        job = next((tmp_path/'calls').glob('codex-*'))
        assert 'thread.started' in (job/'events.jsonl').read_text()
        assert 'fixture transport started' in (job/'stderr.log').read_text()

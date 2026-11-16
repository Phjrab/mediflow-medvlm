"""CPU-only scientific equivalence tests; actual native Phi3 + PEFT modules."""
import copy
import contextlib
from types import SimpleNamespace

import pytest
torch = pytest.importorskip('torch')
pytest.importorskip('peft')
pytest.importorskip('transformers')
from peft import LoraConfig, get_peft_model
from transformers import Phi3Config, Phi3ForCausalLM

from mediflow_medvlm import campaign_gpu_smoke as worker
from mediflow_medvlm import campaign_main_train as main


@pytest.fixture
def model():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(20261023)
    config = Phi3Config(vocab_size=101, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=4096, original_max_position_embeddings=4096,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=False, tie_word_embeddings=True,
        attention_dropout=.1, resid_pdrop=.1, embd_pdrop=.1)
    config._attn_implementation = 'sdpa'
    native = Phi3ForCausalLM(config)
    assert native.lm_head.weight is native.model.embed_tokens.weight
    native.requires_grad_(False)
    native.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    native.enable_input_require_grads()
    targets = [f'model.layers.{i}.{part}.{projection}' for i in range(2)
        for part, projections in [('self_attn', ('qkv_proj', 'o_proj')),
                                 ('mlp', ('gate_up_proj', 'down_proj'))]
        for projection in projections]
    value = get_peft_model(native, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05,
        target_modules=targets, task_type='CAUSAL_LM', bias='none'))
    # Mature adapters make both A/B gradient checks meaningful; separate fresh
    # initialization is used by the actual owner-only GPU probe.
    with torch.no_grad():
        for name, p in value.named_parameters():
            if '.lora_B.' in name: p.normal_(std=.02)
    value.train()
    yield value
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(previous)


def batch(length, sparse=False):
    ids = torch.arange(length).reshape(1, -1) % 95 + 3
    ids[:, -1] = 2  # EOS must be supervised, including the final token.
    labels = ids.clone()
    labels[:, :7] = -100
    if sparse:
        labels[:, 8:-1:3] = -100
    return dict(input_ids=ids, labels=labels, attention_mask=torch.ones_like(ids),
                position_ids=torch.arange(length).reshape(1, -1))


def dense(model, inputs):
    output = model(**{k: v for k, v in inputs.items() if k != 'labels'})
    targets = inputs['labels'][:, 1:]
    mask = targets != -100
    return torch.nn.functional.cross_entropy(output.logits[:, :-1][mask].float(), targets[mask])


def grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}


@pytest.mark.parametrize('length,sparse', [(8, False), (70, False), (71, False), (72, False),
    (17, True), (130, True), (1724, False), (1930, False)])
@pytest.mark.parametrize('autocast', [False, True])
def test_native_loss_hidden_lora_rng_and_eos(model, length, sparse, autocast, record_property):
    inputs = batch(length, sparse)
    native = model.get_base_model()
    hidden = []
    def retain(module, args, output):
        output.last_hidden_state.retain_grad()
        hidden.append(output.last_hidden_state)
    hook = native.model.register_forward_hook(retain)
    cm = lambda: torch.autocast('cpu', dtype=torch.bfloat16) if autocast else contextlib.nullcontext()
    torch.manual_seed(705)
    with cm(): expected = dense(model, inputs)
    (expected / 16).backward()
    expected_grads, expected_hidden = grads(model), hidden[-1].grad.clone()
    expected_rng = torch.get_rng_state().clone()
    model.zero_grad(set_to_none=True)
    hidden.clear()
    torch.manual_seed(705)
    with cm():
        # _target_loss disables CPU autocast intentionally; exercise its exact
        # production helper under CPU BF16 to verify checkpoint replay context.
        actual = worker._phi3_target_loss(native, inputs, inputs['labels'][:, 1:] != -100)
    (actual / 16).backward()
    tol = dict(rtol=3e-2, atol=3e-5) if autocast else dict(rtol=6e-5, atol=2e-7)
    torch.testing.assert_close(actual, expected, rtol=1e-3 if autocast else 2e-6, atol=2e-6)
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert bool(expected_grads[name].abs().sum())
            torch.testing.assert_close(p.grad, expected_grads[name], **tol)
        else: assert p.grad is None
    torch.testing.assert_close(hidden[-1].grad, expected_hidden, **tol)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert native.lm_head.weight.dtype == torch.float32
    assert inputs['labels'][0, -1] == native.config.eos_token_id
    record_property('target_count', int((inputs['labels'][:, 1:] != -100).sum()))
    record_property('loss_abs_diff', float((actual.detach() - expected.detach()).abs()))
    record_property('hidden_grad_max_abs_diff', float((hidden[-1].grad - expected_hidden).abs().max()))
    record_property('lora_grad_max_abs_diff', max(float((p.grad - expected_grads[n]).abs().max())
        for n, p in model.named_parameters() if p.requires_grad))
    hook.remove()


def test_checkpoint_bounds_head_saved_buffers_and_single_decoder(model, monkeypatch):
    from torch.utils import checkpoint as cp
    native = model.get_base_model()
    linear, checkpoint = torch.nn.functional.linear, cp.checkpoint
    rows, chunks, saved, calls = [], [], [], []
    def observe_linear(x, weight, bias=None):
        if weight is native.lm_head.weight: rows.append(x.numel() // x.shape[-1])
        return linear(x, weight, bias)
    def observe_checkpoint(fn, *args, **kwargs):
        if fn.__name__ == 'chunk_sum':
            assert kwargs['use_reentrant'] is False
            chunks.append(args[0].shape[0])
        return checkpoint(fn, *args, **kwargs)
    monkeypatch.setattr(torch.nn.functional, 'linear', observe_linear)
    monkeypatch.setattr(cp, 'checkpoint', observe_checkpoint)
    hook = native.model.register_forward_pre_hook(lambda *args: calls.append(1))
    def pack(x): saved.append(tuple(x.shape)); return x
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        actual, count = worker._target_loss(model, batch(1930))
        # No vocabulary-width activation is saved at all, rather than merely
        # checking the largest chunk of a naively retained CE graph.
        assert not any(len(s) == 2 and s[-1] == 101 for s in saved)
        actual.backward()
    assert count == 1923 and sum(chunks) == count
    assert len(calls) == 1 and max(rows) <= 64 and max(chunks) <= 64
    assert len(rows) >= 2 * len(chunks)  # Replay actually happened.
    hook.remove()


@pytest.mark.parametrize('divisor', [16, 13])
def test_actual_example_divisor_equal_example_weight(model, divisor, record_property):
    # Run the production accumulation function, including its final partial
    # update; compare the unclipped gradients to independent dense averaging.
    inputs = [batch(17 + i * 9, i % 2 == 0) for i in range(divisor)]
    records = [dict(sample_id=str(i)) for i in range(divisor)]
    initial_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1. - step / 500)
    torch.manual_seed(81)
    for value in inputs: (dense(model, value) / divisor).backward()
    expected = grads(model)
    expected_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
    optimizer.step(); scheduler.step()
    expected_state = copy.deepcopy(model.state_dict())
    model.zero_grad(set_to_none=True)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1. - step / 500)
    snapshots = []
    def example(record):
        value = inputs[int(record['sample_id'])]
        loss, count = worker._target_loss(model, value)
        return loss, count, value['input_ids'].numel()
    torch.manual_seed(81)
    metrics = main._updates(model, optimizer, scheduler, records, {}, example,
        lambda: snapshots.append(grads(model)) or {})
    assert metrics['micro_loss_divisor_per_update'] == [divisor]
    assert metrics['sample_ids'] == [str(i) for i in range(divisor)]
    assert len(set(metrics['example_supervised_tokens'])) > 1
    for name, value in expected.items():
        torch.testing.assert_close(snapshots[0][name], value, rtol=8e-5, atol=3e-7)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected_state[name], rtol=8e-5, atol=3e-7)
        if '.lora_' not in name: assert torch.equal(value, initial_state[name])
    assert metrics['final_learning_rate'] == 2e-4 * (1. - 1 / 500)
    assert abs(metrics['gradient_norms_before_clip'][0] - float(expected_norm)) < 1e-6
    record_property('unclipped_lora_grad_max_abs_diff', max(float((snapshots[0][n] - p).abs().max())
        for n, p in expected.items()))


def test_prompt_padding_and_all_target_positions(model):
    value = batch(73, True)
    value['input_ids'][0, -2] = value['labels'][0, -2] = 2
    value['attention_mask'][0, -1] = 0
    value['labels'][0, -1] = -100
    torch.manual_seed(49)
    expected = dense(model, value)
    torch.manual_seed(49)
    actual, count = worker._target_loss(model, value)
    assert count == int((value['labels'][:, 1:] != -100).sum())
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_extreme_finite_full_vocabulary_ce_and_trainable_hidden(model, dtype):
    # Isolate the frozen linear/FP32 CE math from the real decoder/dropout tests.
    # FP64 hidden/head still use FP32 CE, matching the production loss contract.
    native = model.get_base_model()
    native.lm_head.to(dtype=dtype)
    with torch.no_grad(): native.lm_head.weight.mul_(1000.)
    torch.manual_seed(41)
    features = (torch.randn(1, 74, 16, dtype=dtype) * 100.).requires_grad_()
    class Decoder(torch.nn.Module):
        def forward(self, **kwargs): return SimpleNamespace(last_hidden_state=features)
    native.model = Decoder()
    value = batch(74, True)
    mask = value['labels'][:, 1:] != -100
    logits = native.lm_head(features[:, :-1][mask]).float()
    expected = torch.nn.functional.cross_entropy(logits, value['labels'][:, 1:][mask])
    expected.backward()
    expected_grad = features.grad.clone()
    features.grad = None
    actual = worker._phi3_target_loss(native, value, mask)
    actual.backward()
    assert bool(torch.isfinite(actual)) and bool(torch.isfinite(features.grad).all())
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-3)
    torch.testing.assert_close(features.grad, expected_grad, rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize('failure', ['empty', 'batch-two', 'trainable-head', 'hooked-head', 'backward-pre-hook', 'unsupported-class'])
def test_phi3_fail_closed(model, failure):
    inputs = batch(17)
    native = model.get_base_model()
    if failure == 'empty': inputs['labels'].fill_(-100)
    elif failure == 'batch-two': inputs = {k: v.repeat(2, 1) for k, v in inputs.items()}
    elif failure == 'trainable-head': native.lm_head.weight.requires_grad_(True)
    elif failure == 'hooked-head': native.lm_head.register_forward_hook(lambda *args: None)
    elif failure == 'backward-pre-hook': native.lm_head.register_full_backward_pre_hook(lambda *args: None)
    else:
        class Unsupported(Phi3ForCausalLM): pass
        native.__class__ = Unsupported
    with pytest.raises(worker.SmokeError): worker._target_loss(model, inputs)


def test_eval_no_grad_no_checkpoint(model, monkeypatch):
    from torch.utils import checkpoint as cp
    model.eval()
    def forbidden(*args, **kwargs): pytest.fail('no-grad evaluation cannot checkpoint')
    monkeypatch.setattr(cp, 'checkpoint', forbidden)
    with torch.no_grad():
        expected = dense(model, batch(130, True))
        actual, count = worker._target_loss(model, batch(130, True))
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    assert not actual.requires_grad and count == int((batch(130, True)['labels'][:, 1:] != -100).sum())

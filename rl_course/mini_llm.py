"""A real SmolLM2 SFT -> DPO experiment, intentionally small and transparent.

Only the final decoder layer and final normalization are trained. Data are
synthetic sentiment preferences, not collected human feedback. The immutable
SFT reference is represented by cached completion log probabilities for the
fixed preference pairs; it is never overwritten during DPO.
"""
import os
from time import perf_counter
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = 'HuggingFaceTB/SmolLM2-135M'
REVISION = '93efa2f097d58c2a74874c7e644dbc9b0cee75a2'
DATA_VERSION = 'SyntheticSentiment-v1'


def make_data(nouns):
    examples = []
    for label, adjectives in [('positive', ['wonderful', 'excellent', 'lovely', 'great', 'delightful', 'fantastic']),
                              ('negative', ['awful', 'terrible', 'horrible', 'bad', 'disappointing', 'unpleasant'])]:
        for adjective in adjectives:
            for noun in nouns:
                prompt = f'Classify the sentiment as positive or negative.\nReview: The {noun} was {adjective}.\nAnswer:'
                examples.append({'prompt': prompt, 'chosen': ' ' + label,
                                 'rejected': ' ' + ('negative' if label == 'positive' else 'positive')})
    return examples


def load():
    """Downloads public safetensors on first use; no token or remote code needed."""
    cache = os.environ.get('RL_COURSE_MODEL_CACHE')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, cache_dir=cache)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, cache_dir=cache, dtype=torch.float32,
        use_safetensors=True, attn_implementation='eager')
    model.requires_grad_(False)
    model.model.layers[-1].requires_grad_(True)
    model.model.norm.requires_grad_(True)
    model.config.use_cache = False
    # Eval mode disables dropout; it does NOT disable gradients. Keep the
    # same distribution for cached reference likelihoods and policy updates.
    model.eval()
    return model, tokenizer


def batch_tokens(tokenizer, examples, key='chosen'):
    sequences, masks = [], []
    for example in examples:
        prompt = tokenizer.encode(example['prompt'], add_special_tokens=False)
        completion = tokenizer.encode(example[key], add_special_tokens=False) + [tokenizer.eos_token_id]
        sequences.append(prompt + completion)
        masks.append([0] * len(prompt) + [1] * len(completion))
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), tokenizer.pad_token_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    completion_mask = torch.zeros_like(ids, dtype=torch.float32)
    for i, (sequence, mask) in enumerate(zip(sequences, masks)):
        ids[i, :len(sequence)] = torch.tensor(sequence)
        attention[i, :len(sequence)] = 1
        completion_mask[i, :len(sequence)] = torch.tensor(mask)
    return ids, attention, completion_mask


def completion_logps(model, batch):
    ids, attention, completion_mask = batch
    logits = model(input_ids=ids, attention_mask=attention).logits[:, :-1]
    token_logps = logits.log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    # Prediction at index t corresponds to token t+1. Prompt and pad tokens
    # are excluded; EOS is included once. DPO uses sums, not length averages.
    return (token_logps * completion_mask[:, 1:]).sum(-1)


@torch.no_grad()
def pair_scores(model, tokenizer, examples, batch_size=4):
    chosen, rejected = [], []
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        chosen.append(completion_logps(model, batch_tokens(tokenizer, batch, 'chosen')))
        rejected.append(completion_logps(model, batch_tokens(tokenizer, batch, 'rejected')))
    return torch.cat(chosen), torch.cat(rejected)


@torch.no_grad()
def generated_answers(model, tokenizer, examples):
    responses = []
    for example in examples:
        tokens = tokenizer(example['prompt'], return_tensors='pt', add_special_tokens=False)
        output = model.generate(**tokens, max_new_tokens=4, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        responses.append(tokenizer.decode(output[0, tokens.input_ids.shape[1]:], skip_special_tokens=True).strip())
    return responses


def run(seed=7, sft_steps=36, dpo_steps=24):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model, tokenizer = load()
    train = make_data(['film', 'meal', 'book', 'game'])
    test = make_data(['service', 'music'])
    assert not {x['prompt'] for x in train} & {x['prompt'] for x in test}
    params = [p for p in model.parameters() if p.requires_grad]
    parameter_counts = (sum(p.numel() for p in model.parameters()), sum(p.numel() for p in params))
    rows, examples, losses = [], {}, {'SFT': [], 'DPO': []}
    reference_test = None

    def measure(stage, seconds):
        win, lose = pair_scores(model, tokenizer, test)
        responses = generated_answers(model, tokenizer, test)
        exact = np.mean([response.lower() == item['chosen'].strip() for response, item in zip(responses, test)])
        relative = float((win - lose - reference_test).mean()) if reference_test is not None else None
        rows.append({'stage': stage, 'heldout_pair_accuracy': float((win > lose).float().mean()),
                     'heldout_generation_accuracy': float(exact), 'mean_pair_margin': float((win - lose).mean()),
                     'margin_change_from_SFT': relative, 'training_seconds': seconds})
        examples[stage] = responses
        print(rows[-1], flush=True)

    measure('Pretrained', 0.)
    optimizer = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.)
    started = perf_counter()
    for _ in range(sft_steps):
        selected = [train[i] for i in rng.choice(len(train), 4, replace=False)]
        batch = batch_tokens(tokenizer, selected)
        loss = -(completion_logps(model, batch) / batch[2].sum(1)).mean()
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(params, 1.); optimizer.step()
        losses['SFT'].append(float(loss.detach()))
    sft_seconds = perf_counter() - started
    reference_train = tuple(t.clone() for t in pair_scores(model, tokenizer, train))
    test_win, test_lose = pair_scores(model, tokenizer, test)
    reference_test = test_win - test_lose
    measure('SFT', sft_seconds)
    optimizer = torch.optim.AdamW(params, lr=5e-5, weight_decay=0.)
    started = perf_counter()
    for _ in range(dpo_steps):
        indices = rng.choice(len(train), 4, replace=False)
        selected = [train[i] for i in indices]
        win = completion_logps(model, batch_tokens(tokenizer, selected, 'chosen'))
        lose = completion_logps(model, batch_tokens(tokenizer, selected, 'rejected'))
        ref_margin = reference_train[0][indices] - reference_train[1][indices]
        loss = -nn.functional.logsigmoid(.1 * (win - lose - ref_margin)).mean()
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(params, 1.); optimizer.step()
        losses['DPO'].append(float(loss.detach()))
    measure('DPO', perf_counter() - started)
    assert np.isfinite(losses['SFT']).all() and np.isfinite(losses['DPO']).all()
    assert abs(losses['DPO'][0] - np.log(2)) < 1e-4, 'Policy starts at the SFT reference'
    return {'rows': rows, 'examples': examples, 'losses': losses, 'test': test,
            'parameter_counts': parameter_counts, 'model': model, 'tokenizer': tokenizer}

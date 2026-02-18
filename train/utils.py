import trl
import torch
from typing import List, Any, Dict, Union, Optional
from torch.utils.data import SequentialSampler
import re

def add_and_init_special_tokens(model, tokenizer, new_special_tokens: Optional[List[str]] = None):
    """
    Adds new special tokens to the tokenizer and initializes their embeddings.
    """
    if new_special_tokens is None:
        new_special_tokens = [
            "<Think>", "</Think>", "<Plan>", "</Plan>", "<Step>", "</Step>", 
            "<Outline>", "</Outline>", "<Execution>", "</Execution>", "<Conclusion>", "</Conclusion>"
        ]
    
    tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens})
    model.resize_token_embeddings(new_num_tokens=len(tokenizer), pad_to_multiple_of=64)

    embed = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    tied = embed.weight.data_ptr() == lm_head.weight.data_ptr()

    for tok in new_special_tokens:
        base_word = tok.strip("<>")
        base_ids = tokenizer(base_word, add_special_tokens=False).input_ids
        
        if all(i != tokenizer.unk_token_id for i in base_ids):
            avg_embed = embed(torch.tensor(base_ids, device=model.device)).mean(dim=0)
            special_id = tokenizer.convert_tokens_to_ids(tok)
            embed.weight.data[special_id] = avg_embed
            
            if not tied and lm_head.weight.shape == embed.weight.shape:
                avg_lm_logits = lm_head.weight.data[base_ids].mean(dim=0)
                lm_head.weight.data[special_id] = avg_lm_logits.clone()
        else:
            valid_ids = [i for i in base_ids if i != tokenizer.unk_token_id]
            print(f"Warning: Failed to init {tok}, some base tokens are unknown. Using available tokens: {[tokenizer.convert_ids_to_tokens(i) for i in valid_ids]}")
            if valid_ids:
                avg_embed = embed(torch.tensor(valid_ids, device=model.device)).mean(dim=0)
                special_id = tokenizer.convert_tokens_to_ids(tok)
                embed.weight.data[special_id] = avg_embed
                if not tied and lm_head.weight.shape == embed.weight.shape:
                    avg_lm_logits = lm_head.weight.data[valid_ids].mean(dim=0)
                    lm_head.weight.data[special_id] = avg_lm_logits.clone()


TAG_TOKEN_IDS = {
    'plan_start': '<Plan>',
    'plan_end': '</Plan>',
    'outline_start': '<Outline>',
    'outline_end': '</Outline>',
    'execution_start': '<Execution>',
    'execution_end': '</Execution>',
    'step_start': '<Step>',
    'step_end': '</Step>',
    'conclusion_start': '<Conclusion>',
    'conclusion_end': '</Conclusion>',
}

_STEP_RE = re.compile(r"Transient\s+Step\s+(\d+)", re.IGNORECASE)
_DEP_RE  = re.compile(r"Dependency\s*:\s*\[([0-9,\s]*)\]", re.IGNORECASE)

def generate_multiverse_attention_mask(input_ids, tokenizer, device='cpu'):
    seq_len = len(input_ids)
    # Start with a lower triangular matrix (causal mask)
    bool_attention_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)) # Keep bool intermediate mask

    # Assuming single-token tags for simplicity based on original code
    # If tags can be multi-token, this conversion needs adjustment
    outline_start_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS['outline_start'])
    outline_end_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS['outline_end'])
    step_start_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS['step_start'])
    step_end_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS['step_end'])
    #print(path_start_id, path_end_id, parallel_start_id, parallel_end_id)

    deps_map = {}
    structure_stack = []
    path_spans = []
    i = 0
    while i < seq_len:
        current_token_id = input_ids[i]

        # Check <Outline> start
        if current_token_id == outline_start_id:
            structure_stack.append({'type': 'outline', 'start_marker_index': i})
            i += 1
            continue

        # Check <\Outline> start
        elif current_token_id == outline_end_id:
            if not structure_stack or structure_stack[-1]['type'] != 'outline':
                raise ValueError(f"</Outline> found at index {i} without a matching <Outline> block on stack.")
            closed_outline = structure_stack.pop()
            start_idx = closed_outline['start_marker_index']
            inner_start, inner_end = start_idx + 1, i
            text = tokenizer.decode(input_ids[inner_start:inner_end], skip_special_tokens=False) if inner_end > inner_start else ""

            # Parse "Transient Step N"
            sid = None
            m = _STEP_RE.search(text)
            if m:
                sid = int(m.group(1))

            # Parse "Dependency: [ ... ]"
            deps = set()
            m = _DEP_RE.search(text)
            if m:
                raw = (m.group(1) or "").strip()
                if raw:
                    for t in raw.split(","):
                        t = t.strip()
                        if t.isdigit():
                            deps.add(int(t))
            if sid is not None:
                deps_map[sid] = deps
            i += 1
            continue

        # Check <Step> start
        elif current_token_id == step_start_id:
            structure_stack.append({'type': 'step', 'start_marker_index': i})
            i += 1
            continue
            
        # Check </Step> end
        elif current_token_id == step_end_id:
            if not structure_stack or structure_stack[-1]['type'] != 'step':
                raise ValueError(f"</Step> found at index {i} without a matching <Step> block on stack.")
            closed_step = structure_stack.pop()
            start_idx = closed_step['start_marker_index']
            end_excl = i + 1
            inner_start, inner_end = start_idx + 1, i
            step_text = tokenizer.decode(input_ids[inner_start:inner_end], skip_special_tokens=False) if inner_end > inner_start else ""
            m = _STEP_RE.search(step_text)
            sid = int(m.group(1)) if m else None
            deps = deps_map.get(sid, set()) if sid is not None else set()
            
            existing_steps = {s for _, _, s in path_spans if s is not None}
            conflict = len(deps & existing_steps) > 0
            if not conflict:
                path_spans.append((start_idx, end_excl, sid))
            else:
                if len(path_spans) > 1:
                    all_i_indices_to_mask = []
                    all_j_indices_to_mask = []
                    for path_idx_a in range(len(path_spans)):
                        start_a, end_a, _ = path_spans[path_idx_a]
                        # Ensure valid span before creating range
                        if start_a >= end_a:
                            continue
                        indices_a = torch.arange(start_a, end_a, device=device)

                        for path_idx_b in range(path_idx_a + 1, len(path_spans)):
                            start_b, end_b, _ = path_spans[path_idx_b]
                            # Ensure valid span before creating range
                            if start_b >= end_b:
                                continue
                            indices_b = torch.arange(start_b, end_b, device=device)

                            # Use broadcasting to get all (i, j) pairs efficiently
                            grid_i, grid_j = torch.meshgrid(indices_a, indices_b, indexing='ij')

                            all_i_indices_to_mask.append(grid_i.flatten())
                            all_j_indices_to_mask.append(grid_j.flatten())
                        
                    if all_i_indices_to_mask: # Check if there's anything to mask
                        final_i = torch.cat(all_i_indices_to_mask)
                        final_j = torch.cat(all_j_indices_to_mask)

                        # Apply mask using advanced indexing (ensure indices are valid)
                        # For bool mask, False means masked
                        bool_attention_mask[final_i, final_j] = False
                        bool_attention_mask[final_j, final_i] = False # Symmetric masking

                path_spans.clear()
                path_spans.append((start_idx, end_excl, sid))

            i += 1
            continue

        # Move to next token if no tag matched
        i += 1
    # --- End of parsing loop ---

    # Final check for unclosed blocks
    if structure_stack:
        print(structure_stack)
        print(input_ids)
        unclosed_types = [block['type'] for block in structure_stack]
        raise ValueError(f"Input sequence ended with unclosed blocks: {unclosed_types}")

    if len(path_spans) > 1:
        all_i_indices_to_mask = []
        all_j_indices_to_mask = []
        for path_idx_a in range(len(path_spans)):
            start_a, end_a, _ = path_spans[path_idx_a]
            if start_a >= end_a:
                continue
            indices_a = torch.arange(start_a, end_a, device=device)
            for path_idx_b in range(path_idx_a + 1, len(path_spans)):
                start_b, end_b, _ = path_spans[path_idx_b]
                if start_b >= end_b:
                    continue
                indices_b = torch.arange(start_b, end_b, device=device)
                grid_i, grid_j = torch.meshgrid(indices_a, indices_b, indexing='ij')
                all_i_indices_to_mask.append(grid_i.flatten())
                all_j_indices_to_mask.append(grid_j.flatten())     
        if all_i_indices_to_mask:
            final_i = torch.cat(all_i_indices_to_mask)
            final_j = torch.cat(all_j_indices_to_mask)
            bool_attention_mask[final_i, final_j] = False
            bool_attention_mask[final_j, final_i] = False
    path_spans.clear()

    # Convert the final boolean mask to float format (0.0 for True, -inf for False)
    float_attention_mask = torch.full_like(bool_attention_mask, -torch.inf, dtype=torch.float)
    float_attention_mask = float_attention_mask.masked_fill(bool_attention_mask, 0.0)

    return float_attention_mask


def generate_multiverse_position_ids(input_ids: List[int], tokenizer) -> List[int]:
    """
    Make position ids reflect parallel Steps:
      1) Parse <Outline> -> deps_map (step_id -> set(deps)).
      2) Collect <Step> spans in order, split into chunks: a new chunk starts when current step depends on any step in the current chunk.
      3) For each chunk, align all step starts to the same anchor; chunk length = max step length; shift the tail so next chunk starts at anchor+max_len.
    """
    S = len(input_ids)
    position_ids = torch.arange(S, device='cpu', dtype=torch.long)

    outline_start_id = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS['outline_start'])
    outline_end_id   = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS['outline_end'])
    step_start_id    = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS['step_start'])
    step_end_id      = tokenizer.convert_tokens_to_ids(TAG_TOKEN_IDS['step_end'])

    deps_map = {}
    block_stack = []
    steps = []

    i = 0
    while i < S:
        tid = input_ids[i]
        if tid == outline_start_id:
            block_stack.append({'type': 'outline', 'start': i})
            i += 1
            continue
        elif tid == outline_end_id:
            if not block_stack or block_stack[-1]['type'] != 'outline':
                raise ValueError(f"</Outline> at {i} without a matching <Outline>.")
            blk = block_stack.pop()
            st = blk['start']
            txt = tokenizer.decode(input_ids[st+1:i], skip_special_tokens=False) if i > st+1 else ""
            sid = None
            m = _STEP_RE.search(txt)
            if m:
                sid = int(m.group(1))
            # deps
            deps = set()
            m = _DEP_RE.search(txt)
            if m:
                raw = (m.group(1) or "").strip()
                if raw:
                    for t in raw.split(","):
                        t = t.strip()
                        if t.isdigit():
                            deps.add(int(t))
            if sid is not None:
                deps_map[sid] = deps
            i += 1
            continue
        elif tid == step_start_id:
            block_stack.append({'type': 'step', 'start': i})
            i += 1
            continue
        elif tid == step_end_id:
            if not block_stack or block_stack[-1]['type'] != 'step':
                raise ValueError(f"</Step> at {i} without a matching <Step>.")
            blk = block_stack.pop()
            st = blk['start']
            ed = i + 1  # include </Step>
            txt = tokenizer.decode(input_ids[st+1:i], skip_special_tokens=False) if i > st+1 else ""
            m = _STEP_RE.search(txt)
            sid = int(m.group(1)) if m else None
            steps.append((st, ed, sid))
            i = ed
            continue
        i += 1

    if block_stack:
        raise ValueError(f"Unclosed blocks: {[b['type'] for b in block_stack]}")
    if not steps:
        return position_ids

    chunks = []
    cur_chunk, cur_ids = [], set()

    for (st, ed, sid) in steps:
        deps = deps_map.get(sid, set()) if sid is not None else set()
        conflict = bool(cur_chunk) and len(deps & cur_ids) > 0
        if conflict:
            chunks.append(cur_chunk)
            cur_chunk, cur_ids = [], set()
        cur_chunk.append((st, ed, sid))
        if sid is not None:
            cur_ids.add(sid)

    if cur_chunk:
        chunks.append(cur_chunk)

    offset = torch.zeros(S, dtype=torch.long)
    running_base = min(st for (st, ed, sid) in chunks[0])

    for chunk in chunks:
        smin = min(st for (st, ed, sid) in chunk)
        emax = max(ed for (st, ed, sid) in chunk)
        max_len = max(ed - st for (st, ed, sid) in chunk)

        for (st, ed, sid) in chunk:
            shift = running_base - st
            if shift != 0 and ed > st:
                offset[st:ed] += shift

        tail_delta = (running_base + max_len) - emax
        if tail_delta != 0 and emax < S:
            offset[emax:] += tail_delta

        running_base = running_base + max_len

    position_ids = position_ids + offset

    if position_ids.numel() != S:
        raise ValueError("Position ID generation length mismatch!")

    return position_ids


class MultiverseDataCollatorForCompletionOnlyLM(trl.DataCollatorForCompletionOnlyLM):
    def __init__(self, *args, max_length=None, use_multiverse: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_length = max_length
        self.use_multiverse = use_multiverse
    
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        if not self.use_multiverse:
            return super().torch_call(examples)

        # First, generate full attention masks and position ids for complete sequences
        attention_masks = []
        position_ids = []
        
        for example in examples:
            # Get the complete input_ids (before any truncation)
            if isinstance(example, dict):
                input_ids = example['input_ids']
            else:
                input_ids = example
            
            # Generate full attention mask and position ids based on complete sequence
            attention_mask = generate_multiverse_attention_mask(input_ids, self.tokenizer)
            position_id = generate_multiverse_position_ids(input_ids, self.tokenizer)
            
            attention_masks.append(attention_mask)
            position_ids.append(position_id)
        
        # Apply the standard collation with truncated examples
        batch = super().torch_call(examples)
        
        # Get the final sequence length after truncation
        final_seq_len = batch['input_ids'].shape[1]
        
        # Create custom attention masks and position ids with the same truncation
        batch['attention_mask'] = torch.zeros(len(examples), 1, final_seq_len, final_seq_len, dtype=torch.float, device='cpu')
        batch['position_ids'] = torch.zeros(len(examples), final_seq_len, dtype=torch.long, device='cpu')

        for i in range(len(examples)):
            # Apply the same truncation to attention mask and position ids
            batch['attention_mask'][i, 0] = attention_masks[i][:final_seq_len, :final_seq_len]
            batch['position_ids'][i] = position_ids[i][:final_seq_len]
            batch['input_ids'][i] = batch['input_ids'][i][:final_seq_len]
            batch['labels'][i] = batch['labels'][i][:final_seq_len]
        
        return batch


class SequentialSFTTrainer(trl.SFTTrainer):
    """
    Custom SFTTrainer that uses sequential sampling instead of random sampling
    """
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """Override sampler method to use sequential sampling instead of random sampling"""
        if self.train_dataset is None or not hasattr(self.train_dataset, '__len__'):
            return None
        
        # If group_by_length is set, still use length-grouped sampler
        if self.args.group_by_length:
            return super()._get_train_sampler()
        else:
            # Use sequential sampler
            return SequentialSampler(self.train_dataset)
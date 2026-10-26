"""Explicit evaluation-only v2 prompts; never alter a training sample or v1 protocol."""
from __future__ import annotations
from pathlib import Path
import re
import unicodedata
from ..runtime import atomic_json, canonical_hash
from ..training.rendering import image_paths, native_encode, user_text

V2_PROTOCOL = 'medvlm.eval.v2'
VQA_PROMPT = 'medical-vqa-short-answer.v2'
V2_RENDERER = 'medvlm.eval-native.v2'
VQA_INSTRUCTION = ('Answer with only the shortest phrase that directly answers the question. '
                   'Do not describe the image or explain your reasoning. '
                   'For a yes/no question, output only yes or no.')


def evaluation_text(sample, prompt_policy):
    if prompt_policy == VQA_PROMPT and sample['task_type'] == 'medical_vqa':
        # Do not inspect answer_type, source metadata, supervision, or gold.
        restricted = {k: sample[k] for k in ('sample_id', 'task_type', 'input')}
        user_text(restricted)  # Reuse the input allowlist and required-question validation.
        return 'Question:\n'+str(sample['input']['question']).strip()+'\n\n'+VQA_INSTRUCTION
    if prompt_policy in (VQA_PROMPT, 'native-task-instructions.v1'):
        return user_text(sample)
    raise ValueError('unsupported v2 prompt policy')


def encode_evaluation_sample(sample, processor, model_config=None, *, max_length=2048,
                             image_root=None, include_labels=False, history=None,
                             prompt_policy=VQA_PROMPT, image_condition='image'):
    if include_labels or history:
        raise ValueError('v2 evaluator cannot create training labels or conversation history')
    restricted={k:sample[k] for k in ('sample_id','task_type','input')}
    text=evaluation_text(restricted,prompt_policy)
    paths, metadata = image_paths(restricted,image_root) if image_condition=='image' else ([],[])
    if image_condition not in ('image','no_image'):
        raise ValueError('unknown image condition')
    if image_condition=='no_image' and sample['task_type']!='medical_vqa':
        raise ValueError('no-image condition is only defined for VQA')
    opened=[]
    if paths:
        from PIL import Image
        for path,item in zip(paths,metadata):
            with Image.open(path) as original:
                item.update(width=original.width,height=original.height)
                opened.append(original.convert('RGB'))
    content=[{'type':'image','image':str(p)} for p in paths]+[{'type':'text','text':text}]
    # Native text-only models expect string messages, VLMs accept content blocks.
    messages=[{'role':'user','content':content if paths or image_condition=='no_image' else text}]
    encoded=native_encode(processor,messages,generation_prompt=True,images=opened)
    if encoded['input_ids'].shape[-1]>max_length:
        raise ValueError('CONTEXT_OVERFLOW')
    pixels=[k for k in encoded if any(x in k.casefold() for x in ('pixel','image','vision','video'))]
    if image_condition=='no_image' and pixels:
        raise ValueError('UNEXPECTED_NO_IMAGE_TENSORS')
    if image_condition=='no_image' and 'mm_token_type_ids' in encoded:
        mask=encoded['mm_token_type_ids']
        if mask.shape != encoded['input_ids'].shape or not bool((mask==0).all()):
            raise ValueError('NON_TEXT_NO_IMAGE_MODALITY_MASK')
    return encoded,{'sample_id':sample['sample_id'],'images':metadata,'input_hash':canonical_hash(sample['input']),
                    'renderer_version':V2_RENDERER,'encoded_text_sha256':canonical_hash(text)}


def normalize_vqa(text):
    # Terminal punctuation only; retain internal decimal points, units, numbers, negation.
    value=' '.join(unicodedata.normalize('NFKC',str(text)).casefold().split())
    return re.sub(r'[.!?。！？]+$', '',value).strip()


def generation_finish(token_count, last_token_id, maximum, eos_token_ids):
    """A real EOS on the budget boundary is a normal stop, not truncation."""
    if isinstance(eos_token_ids,int):eos_token_ids=[eos_token_ids]
    if last_token_id in (eos_token_ids or []):return 'stop'
    return 'length' if token_count>=maximum else 'stop'


def promote_protocol(protocol, output, *, prompt_policy, max_vqa_tokens=128,
                     validation_evidence=None, previously_observed_test=True):
    if type(max_vqa_tokens) is not int or max_vqa_tokens not in (128,256):
        raise ValueError('declared v2 VQA budget must be 128 or 256')
    if prompt_policy not in (VQA_PROMPT,'native-task-instructions.v1'):
        raise ValueError('unknown prompt policy')
    result={k:v for k,v in protocol.items() if k!='protocol_hash'}
    result.update(protocol_version=V2_PROTOCOL,prompt_policy=prompt_policy,
                  normalization_policy='nfkc-casefold-whitespace-terminal-punctuation.v2',
                  validation_evidence=validation_evidence,
                  previously_observed_test=bool(previously_observed_test),
                  clinical_review='not_performed')
    result['max_new_tokens']={**protocol['max_new_tokens'],'medical_vqa':max_vqa_tokens}
    result['protocol_hash']=canonical_hash(result)
    destination=Path(output)
    if destination.exists():
        import json
        if json.loads(destination.read_text())!=result:raise ValueError('refusing to overwrite frozen v2 protocol')
    else:atomic_json(destination,result)
    return result

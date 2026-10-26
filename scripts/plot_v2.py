#!/usr/bin/env python3
"""Plot verified aggregate v2 results only; never read predictions or source text."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

VLM_MODELS = ('qwen35_2b', 'medgemma15_4b')
TEXT_MODELS = ('qwen3_17b', 'qwen3_4b', 'medgemma15_4b')
MODEL_LABELS = {'qwen35_2b': 'Qwen3.5 2B', 'medgemma15_4b': 'MedGemma 1.5 4B',
                'qwen3_17b': 'Qwen3 1.7B', 'qwen3_4b': 'Qwen3 4B'}
CONDITION_LABELS = {'base': 'Base', 'tuned_v1': 'v1 adapter', 'en_v1': 'EN v1', 'ko': 'KO v2', 'mix': 'MIX v2'}
COLORS = {'base': '#486A88', 'tuned_v1': '#C8843B', 'en_v1': '#9AA8B2', 'ko': '#348B79', 'mix': '#C8843B'}
VQA_SOURCES = ('slake', 'vqa_rad')
VQA_GROUPS = ('ALL', 'OPEN', 'CLOSED')
TEXT_METRICS = ('kor_accuracy', 'kor_coverage', 'aihub_mcq_accuracy')


def point(value, support, protocol_hash):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('verified aggregate metric must be a finite fraction in [0,1]')
    if type(support) is not int or support < 1:
        raise ValueError('verified aggregate support must be a positive integer')
    if not isinstance(protocol_hash, str) or len(protocol_hash) != 64:
        raise ValueError('verified aggregate has no protocol SHA256')
    return {'value': float(value), 'support': support, 'protocol_hash': protocol_hash}


def completed(summary):
    if summary.get('schema_version') != 'medvlm.public-report.v2':
        raise ValueError('expected a v2 public run_summary.json')
    return [cell for cell in summary.get('evaluations', [])
            if cell.get('verified') is True and cell.get('status') == 'SUCCEEDED']


def put(target, key, value):
    if key in target:
        raise ValueError('ambiguous duplicate verified model/condition/cohort')
    target[key] = value


def select_vqa(summary):
    """Image-present source/subgroup EM only. Missing cells remain absent, never zero."""
    selected = {}
    for cell in completed(summary):
        source = {'slake': 'slake', 'vqarad': 'vqa_rad'}.get(cell.get('phase'))
        if (source is None or cell.get('model') not in VLM_MODELS or cell.get('condition') not in ('base','tuned_v1')
                or cell.get('image_condition') != 'image'):
            continue
        for metric in (cell.get('metrics') or {}).get('vqa', []):
            group = metric.get('answer_type')
            if metric.get('source_id') != source or group not in VQA_GROUPS:
                continue
            put(selected, (cell['model'], source, group, cell['condition']),
                point(metric['v2_normalized_lexical_em'], metric['scheduled'], cell.get('protocol_hash')))
    for source in VQA_SOURCES:
        if len({v['protocol_hash'] for k,v in selected.items() if k[1] == source}) > 1:
            raise ValueError('VQA comparison requires one matched protocol per source')
    return selected


def select_text(summary):
    """Keep KorMedMCQA label accuracy/coverage and AI-Hub q_type=1 MCQ separate."""
    selected = {}
    for cell in completed(summary):
        model, condition = cell.get('model'), cell.get('condition')
        if cell.get('phase') != 'text' or model not in TEXT_MODELS or condition not in ('base','en_v1','ko','mix'):
            continue
        if model == 'medgemma15_4b' and condition != 'base':
            continue
        metrics = cell.get('metrics') or {}; korean = metrics.get('kormedmcqa')
        if korean:
            for name, field in (('kor_accuracy','micro_end_to_end_correct_fraction'), ('kor_coverage','coverage')):
                put(selected, (model, condition, name), point(korean[field], korean['scheduled'], cell.get('protocol_hash')))
        for metric in metrics.get('aihub_q_types', []):
            if str(metric.get('q_type')) == '1' and metric.get('task_type') == 'medical_mcq':
                put(selected, (model, condition, 'aihub_mcq_accuracy'),
                    point(metric['end_to_end_correct_fraction'], metric['scheduled'], cell.get('protocol_hash')))
    if len({v['protocol_hash'] for v in selected.values()}) > 1:
        raise ValueError('text comparison requires one matched protocol')
    return selected


def axes_style(ax):
    ax.set_ylim(0, 116)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#DFE4E7', linewidth=.7)
    ax.tick_params(axis='both', length=0, labelsize=10, pad=7)
    for side in ('top','right','left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#BBC5CB')


def draw_cell(ax, x, datum, color, width=.32):
    if datum is None:
        ax.text(x, 7, 'Incomplete', ha='center', va='bottom', rotation=90,
                fontsize=8, color='#6F777E')
        return
    height = 100 * datum['value']
    ax.bar(x, height, width=width, color=color, zorder=3)
    ax.text(x, height + 2, f"{height:.1f}%\nn={datum['support']:,}",
            ha='center', va='bottom', fontsize=8.5, color='#24333E', linespacing=1.3)


def save_figure(fig, output, name, metadata):
    result = []
    for suffix in ('png','svg'):
        path = output / (name + '.' + suffix)
        if path.is_symlink():
            raise ValueError('refusing to replace a symlinked plot')
        fig.savefig(path, dpi=180, facecolor='white', metadata=metadata)
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
        result.append(path.name)
    return result


def render(summary, output):
    """Use only the passed public aggregate object, plus local plotting/font libraries."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'svg.fonttype':'none',
                         'axes.titleweight':'bold', 'axes.labelcolor':'#24333E', 'text.color':'#24333E'})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    vqa = select_vqa(summary); text = select_text(summary)
    captured = str(summary.get('generated_at', 'unknown snapshot'))
    files = []
    figure, axes = plt.subplots(2, 3, figsize=(14, 8.4), sharey=True)
    for row, source in enumerate(VQA_SOURCES):
        for column, group in enumerate(VQA_GROUPS):
            ax = axes[row, column]; axes_style(ax)
            for index, model in enumerate(VLM_MODELS):
                for condition, offset in (('base', -.19), ('tuned_v1', .19)):
                    draw_cell(ax, index + offset, vqa.get((model, source, group, condition)), COLORS[condition])
            ax.set_xticks([0,1], [MODEL_LABELS[m] for m in VLM_MODELS])
            ax.set_xlim(-.65, 1.65)
            ax.set_title(('SLAKE' if source == 'slake' else 'VQA-RAD') + ' · ' + ('Overall' if group == 'ALL' else group), pad=12)
            if column == 0:
                ax.set_ylabel('Lexical exact match (%)', labelpad=12)
    figure.suptitle('VQA: Base and preserved v1 adapter', fontsize=20, x=.055, ha='left', y=.975)
    figure.text(.055, .92, 'Protocol v2 · image present · normalized EM · support shown above each completed bar', fontsize=11)
    figure.legend(handles=[Patch(color=COLORS[c], label=CONDITION_LABELS[c]) for c in ('base','tuned_v1')],
                  loc='upper right', bbox_to_anchor=(.965,.977), ncol=2, frameon=False)
    figure.text(.055,.083,'MedGemma has prior dataset-level SLAKE training exposure; exact test exposure remains unknown.',fontsize=9)
    figure.text(.055,.057,'Seed 20261023. Previously observed v1 tests are included; not a new blind benchmark. One seed; no clinical accuracy claim.',fontsize=9)
    figure.text(.055,.031,'EM keeps numbers, units and negation. Blank positions marked Incomplete are not zero. Snapshot: '+captured,fontsize=9)
    figure.subplots_adjust(left=.075,right=.975,bottom=.16,top=.83,hspace=.42,wspace=.19)
    files += save_figure(figure, output, 'vqa_v2', {'Title':'VQA protocol v2 aggregate lexical exact match', 'Description':'Verified completed public aggregates only. No clinical accuracy claim.'})
    plt.close(figure)
    if text:
        figure, axes = plt.subplots(3, 3, figsize=(14, 10.5), sharey=True)
        titles = {'kor_accuracy':'KorMedMCQA · micro accuracy', 'kor_coverage':'KorMedMCQA · valid-answer coverage',
                  'aihub_mcq_accuracy':'AI-Hub 71875 · MCQ accuracy'}
        for row, model in enumerate(TEXT_MODELS):
            conditions = ('base',) if model == 'medgemma15_4b' else ('base','en_v1','ko','mix')
            for column, metric in enumerate(TEXT_METRICS):
                ax = axes[row,column]; axes_style(ax)
                for index, condition in enumerate(conditions):
                    draw_cell(ax,index,text.get((model,condition,metric)),COLORS[condition],width=.63)
                ax.set_xticks(list(range(len(conditions))),[CONDITION_LABELS[c] for c in conditions])
                ax.set_xlim(-.65,len(conditions)-.35)
                if column == 0:
                    ax.set_ylabel(MODEL_LABELS[model]+'\nPercent (%)',labelpad=12)
                if row == 0:
                    ax.set_title(titles[metric],fontsize=11,pad=12)
        figure.suptitle('Text QA: matched protocol, distinct answer-label metrics',fontsize=19,x=.055,ha='left',y=.975)
        figure.text(.055,.932,'Protocol v2 · full scheduled denominator · n = source support · MedGemma uses the original text-only Base condition',fontsize=10.5)
        figure.text(.055,.059,'Seed 20261023. One pass, one seed; KO/MIX differ in training exposure. Previously seen v1 tests; pretraining exposure unknown.',fontsize=9)
        figure.text(.055,.038,'MCQ accuracy and coverage are not clinical accuracy. AI-Hub short/long lexical metrics are excluded from this MCQ figure.',fontsize=9)
        figure.text(.055,.017,'Incomplete means not yet verified and scored. Snapshot: '+captured,fontsize=9)
        figure.subplots_adjust(left=.09,right=.975,bottom=.13,top=.86,hspace=.40,wspace=.18)
        files += save_figure(figure,output,'text_v2',{'Title':'Text protocol v2 aggregate label accuracy and coverage','Description':'Verified completed public aggregates only. No clinical accuracy claim.'})
        plt.close(figure)
    return {'files':files,'vqa_completed_metric_cells':len(vqa),'vqa_planned_metric_cells':24,
            'text_completed_metric_cells':len(text),'text_planned_metric_cells':27,
            'text_plot_status':'GENERATED' if text else 'NOT_READY_NO_VERIFIED_TEXT_RESULTS',
            'protocol_hashes':sorted({p['protocol_hash'] for p in [*vqa.values(),*text.values()]}),
            'raw_predictions_or_source_text_read':False,'clinical_accuracy':None}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,default=Path('reports/public_summary/v2/run_summary.json'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raw=args.summary.read_bytes(); summary=json.loads(raw)
    receipt=render(summary,args.output)
    receipt.update(schema_version='medvlm.aggregate-plots.v2',summary_sha256=hashlib.sha256(raw).hexdigest(),
                   report_generated_at=summary.get('generated_at'),seed=20261023)
    destination=args.output/'plot_receipt.json'
    if destination.is_symlink(): raise ValueError('refusing to replace a symlinked receipt')
    destination.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':
    main()

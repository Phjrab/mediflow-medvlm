#!/usr/bin/env python3
"""Descriptive figures from verified aggregate v4 results only."""
import argparse
import json
from pathlib import Path


def plot(summary_path, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    data=json.loads(Path(summary_path).read_text())
    if data['status']!='SUCCEEDED':
        raise ValueError('Verified completed results required')
    output.mkdir(parents=True,exist_ok=True)
    models=[('qwen3_17b','Qwen3-1.7B'),('qwen3_4b','Qwen3-4B')]
    fig, axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for axis,(model,title) in zip(axes,models):
        base=data['reused_base_controls'][model]
        tuned=data['evaluations'][f'{model}-common-tuned-v4-20261112']
        values=[]
        for entry in (base,tuned):
            values.append([entry['korean_summary']['micro_end_to_end_correct_fraction']*100]+[
                next(c['end_to_end_correct_fraction']*100 for c in entry['cohorts']
                     if c['source_id']==source and c['task_type']==task)
                for source,task in [('aihub_71875','medical_mcq'),('pubmedqa_labeled','context_qa')]])
        x=np.arange(3)
        for offset,values_,label,color in [(-.18,values[0],'Original Base','#64748b'),(.18,values[1],'71762 SFT','#087e8b')]:
            bars=axis.bar(x+offset,values_,width=.34,label=label,color=color)
            axis.bar_label(bars,fmt='%.1f',padding=3,fontsize=8)
        axis.set_xticks(x,['KorMedMCQA\nn=3009','AI-Hub MCQ\nn=777','PubMedQA\nn=500'])
        axis.set_title(title)
        axis.set_ylim(0,100)
        axis.spines[['top','right']].set_visible(False)
        axis.grid(axis='y',alpha=.2)
        axis.set_axisbelow(True)
    axes[0].set_ylabel('Correct / all scheduled (%)')
    axes[1].legend(frameon=False,fontsize=8,loc='upper left')
    fig.suptitle('Common benchmark retention after category-random SFT',fontsize=12)
    fig.text(.5,.015,'Single seed; previously observed benchmarks. Training-token exposure differs from earlier SFT conditions.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.94))
    for extension in ('png','svg'):
        fig.savefig(output/('benchmark_retention.'+extension),dpi=220)
    plt.close(fig)
    fig,axis=plt.subplots(figsize=(6,3.8))
    x=np.arange(2)
    for offset,condition,label,color in [(-.18,'base','Original Base','#64748b'),(.18,'tuned','71762 SFT','#087e8b')]:
        values=[data['evaluations'][f'{model}-random-{condition}-v4-20261112']['cohorts'][0]['lexical_token_f1']
                for model,_ in models]
        bars=axis.bar(x+offset,values,width=.34,label=label,color=color)
        axis.bar_label(bars,fmt='%.3f',padding=3)
    axis.set_xticks(x,[title for _,title in models]);axis.set_ylabel('Lexical token F1')
    axis.set_title('Agreement with randomly assigned references (n=256)')
    axis.set_ylim(0,max(.3,axis.get_ylim()[1]*1.2));axis.legend(frameon=False)
    axis.spines[['top','right']].set_visible(False)
    fig.text(.5,.02,'Descriptive text overlap; not clinical correctness or verified question-answer alignment.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1))
    for extension in ('png','svg'):
        fig.savefig(output/('random_reference_agreement.'+extension),dpi=220)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();plot(args.summary,args.output)

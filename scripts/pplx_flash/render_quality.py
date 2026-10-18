import argparse
import html
import json
from pathlib import Path

MODES=['fp16-scan','int8-tuned','q4-fused']
NAMES=['FP16','Q8','Q4']


def render(result,corpus):
    esc=lambda x:html.escape(str(x),quote=True)
    pct=lambda x:'—' if x is None else f'{100*x:.1f}%'
    cases={c['id']:c for c in corpus['cases']}
    parts=['''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Q4 quality evaluation</title>
<style>body{font:16px/1.5 system-ui;max-width:1150px;margin:32px auto;padding:0 20px;color:#19342c;background:#f5f7f5}h1,h2{line-height:1.2}table{border-collapse:collapse;width:100%;background:white;margin:16px 0}th,td{padding:10px;border-bottom:1px solid #dae3dc;text-align:right}th:first-child,td:first-child{text-align:left}small,.note{color:#52665c}details.case{background:white;border:1px solid #d4e0d7;border-radius:8px;margin:10px 0;padding:12px}summary{cursor:pointer}details.error{border-left:5px solid #b44030}.content{display:grid;grid-template-columns:1fr 1fr;gap:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.4 ui-monospace}input,select{font:inherit;padding:7px}label{display:inline-block;margin:8px 15px 8px 0}input[type=text]{width:300px} .probabilities{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}@media(max-width:700px){.content,.probabilities{display:block}th,td{padding:5px}}</style></head><body>
<h1>Q4 quality evaluation</h1>''']
    parts.append(f"<p><strong>{result['matched_inputs']} matched inputs</strong> across FP16, Q8 and Q4. Q4 uses symmetric per-output-channel W4A8 quantization; grouped/calibrated Q4 is not tested here.</p>")
    if not result['complete']:parts.append('<p><strong>Partial results. The evaluation is still running.</strong></p>')
    parts.append('<p class="note">Frozen, stratified public test subsets plus authored controls. This is not a full public benchmark or a representative production sample. One primary question per input is used for overall accuracy. No latency claims are made from these quality runs.</p>')
    parts.append('<table><tr><th>Primary task</th><th>N</th><th>FP16 correct</th><th>Q8 correct</th><th>Q4 correct</th></tr>')
    groups=[('All primary questions',result['primary_metrics'])]
    for key,label in [('emotion/emotion','TweetEval emotion'),('sentiment/sentiment','TweetEval sentiment'),('banking/intent','BANKING77: eight-intent subset'),('controlled-tone/tone','Controlled tone'),('controlled-tone/anger','Controlled anger detection (auxiliary)'),('stress/refund','Refund/negation stress cases')]:
        if key in result['groups']:groups.append((label,result['groups'][key]['metrics']))
    for label,metrics in groups:
        parts.append('<tr><td>'+esc(label)+'</td><td>'+str(metrics[MODES[0]]['n'])+'</td>'+''.join(f"<td>{metrics[m]['correct']} / {pct(metrics[m]['accuracy'])}</td>" for m in MODES)+'</tr>')
    parts.append('</table>')
    q4=result['primary_comparisons'][1];ci=q4.get('paired_bootstrap_95_ci')
    parts.append(f"<p>Q4 changed <strong>{q4['decision_flips']}</strong> primary decisions relative to FP16: <strong>{q4['correct_to_wrong']}</strong> correct → wrong and <strong>{q4['wrong_to_correct']}</strong> wrong → correct. <strong>{q4['high_confidence_reference_flips']}</strong> flips had FP16 probability at least 0.90.</p>")
    if ci:parts.append(f"<p class=note>Accuracy difference Q4 − FP16: {100*q4['accuracy_delta']:+.2f} percentage points; paired bootstrap 95% interval [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]. Small subsets limit precision.</p>")
    parts.append('<h2>Score and probability drift</h2><table><tr><th>Compared with FP16</th><th>Score mean absolute change</th><th>Score maximum change</th><th>Binary mean absolute change</th><th>Binary maximum change</th></tr>')
    for comp,name in zip(result['all_question_drift'],['Q8','Q4']):
        score=comp['numeric_drift'].get('score',{});binary=comp['numeric_drift'].get('noul',{})
        parts.append('<tr><td>'+name+'</td>'+''.join(f'<td>{x:.4f}</td>' for x in [score.get('mean_absolute',0),score.get('max_absolute',0),binary.get('mean_absolute',0),binary.get('max_absolute',0)])+'</tr>')
    parts.append('</table><p class=note>Score changes use the 0–2 ordinal scale; binary changes use probability units. A changed output is not automatically a wrong answer. The original multilingual tone example has no authoritative gold label.</p>')
    parts.append('<h2>Inspect individual decisions</h2><label>Search <input id="search" type="text" placeholder="Input text, case ID, or label"></label><label>Task <select id="family"><option value="">All</option>')
    for family in sorted({d['family'] for d in result['details']}):parts.append(f'<option value="{esc(family)}">{esc(family)}</option>')
    parts.append('</select></label><label><input id="primary" type="checkbox" checked> Primary questions only</label><label><input id="errors" type="checkbox"> FP16 correct → Q4 wrong only</label><p id="count"></p>')
    for d in sorted(result['details'],key=lambda d:d['q4_total_variation_vs_fp16'],reverse=True):
        c=cases[d['id']];gold=d['gold'];pred=d['predictions'];error=gold is not None and pred['fp16-scan']==gold and pred['q4-fused']!=gold
        state=c['request']['state'];state=state if isinstance(state,str) else json.dumps(state,ensure_ascii=False,indent=2)
        parts.append(f'<details class="case{" error" if error else ""}" data-family="{esc(d["family"])}" data-primary="{int(d["primary"])}" data-error="{int(error)}"><summary><strong>{esc(d["id"])} / {esc(d["question"])}</strong> · gold: {esc(gold if gold is not None else "unlabeled")} · '+esc(' / '.join(name+': '+pred[mode] for mode,name in zip(MODES,NAMES)))+'</summary>')
        parts.append('<div class="content"><div><h3>Input</h3><pre>'+esc(state)+'</pre><small>'+esc(json.dumps(c['source'],ensure_ascii=False))+'</small></div><div><h3>Question</h3><pre>'+esc(json.dumps(c['request']['questions'][d['question']],ensure_ascii=False,indent=2))+'</pre></div></div><div class="probabilities">')
        for mode,name in zip(MODES,NAMES):
            parts.append('<div><h3>'+name+'</h3><pre>'+esc(json.dumps({'prediction':pred[mode],'score_or_probability':d['numeric'][mode],'probabilities':d['probabilities'][mode]},indent=2,ensure_ascii=False))+'</pre></div>')
        parts.append('</div></details>')
    parts.append('<h2>Reproducibility</h2><pre>'+esc(json.dumps({k:result[k] for k in ['corpus_sha256','wasm_sha256','checkpoint_revision','seed']},indent=2))+'</pre>')
    parts.append('''<script>
const $=id=>document.getElementById(id),rows=[...document.querySelectorAll('details.case')];
function filter(){let shown=0;const text=$('search').value.toLowerCase(),family=$('family').value;for(const row of rows){const visible=(!text||row.textContent.toLowerCase().includes(text))&&(!family||row.dataset.family===family)&&(!$('primary').checked||row.dataset.primary==='1')&&(!$('errors').checked||row.dataset.error==='1');row.hidden=!visible;if(visible)shown++;}$('count').textContent=shown+' decisions shown.';}
for(const id of ['search','family','primary','errors'])$(id).addEventListener('input',filter);filter();
</script></body></html>''')
    return '\n'.join(parts)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with a.output.open('x') as stream:stream.write(render(json.loads(a.analysis.read_text()),json.loads(a.corpus.read_text())))

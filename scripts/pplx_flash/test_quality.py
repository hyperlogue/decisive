import math
import copy
import hashlib
import json
from types import SimpleNamespace
import unittest
from analyze_quality import MODES, analyze, distribution, metric, paired, prediction
from render_quality import render


def score(p):return {'type':'score','probabilities':{str(i):v for i,v in enumerate(p)},'score':sum(i*v for i,v in enumerate(p))}


def row(id='example',gold='1'):
    return {'id':id,'gold':gold,'ordinal':int(gold) if gold is not None else None,
            'answers':dict(zip(MODES,[score([.02,.95,.03]),score([.1,.8,.1]),score([.9,.08,.02])]))}


class QualityMetrics(unittest.TestCase):
    def test_score_category_flip_is_a_decision_error(self):
        r=row();result=paired([r],'q4-fused')
        self.assertEqual(result['decision_flips'],1)
        self.assertEqual(result['correct_to_wrong'],1)
        self.assertEqual(result['high_confidence_reference_flips'],1)
        self.assertEqual(metric([r],'fp16-scan')['accuracy'],1)
        self.assertEqual(metric([r],'q4-fused')['accuracy'],0)
        self.assertAlmostEqual(metric([r],'q4-fused')['ordinal_mae'],.88)
        self.assertAlmostEqual(metric([r],'fp16-scan')['nll'],-math.log(.95))

    def test_changes_can_fix_reference_errors(self):
        a=row();b=row('other','0')
        result=paired([a,b],'q4-fused',bootstrap=True)
        self.assertEqual(result['correct_to_wrong'],1)
        self.assertEqual(result['wrong_to_correct'],1)
        self.assertEqual(result['accuracy_delta'],0)
        self.assertEqual(result['mcnemar_exact_p'],1)
        self.assertEqual(result['decision_flips'],2)

    def test_unlabeled_tone_is_disagreement_not_accuracy(self):
        r=row(gold=None);result=paired([r],'q4-fused')
        self.assertEqual(result['decision_flips'],1)
        self.assertEqual(result['labeled_questions'],0)
        self.assertEqual(result['correct_to_wrong'],0)
        self.assertIsNone(result['accuracy_delta'])
        self.assertIsNone(metric([r],'q4-fused')['accuracy'])

    def test_binary_threshold_crossing_counts(self):
        r=row();r['gold']='true';r['ordinal']=None
        r['answers']={m:{'type':'noul','noul':.4999 if m=='q4-fused' else .5001} for m in MODES}
        result=paired([r],'q4-fused')
        self.assertEqual(result['correct_to_wrong'],1)
        self.assertAlmostEqual(result['numeric_drift']['noul']['mean_absolute'],.0002)

    def test_invalid_probability_and_score_reports_rejected(self):
        for p in [[.1,.1,.1],[float('nan'),.5,.5],[-.1,.5,.6]]:
            with self.assertRaises(AssertionError):distribution(score(p))
        answer=score([.1,.8,.1]);answer['score']=0
        with self.assertRaises(AssertionError):prediction(answer)
        with self.assertRaises(AssertionError):prediction({'type':'choice','probabilities':{'a':.9,'b':.1},'choice':'b'})

    def test_pooled_auxiliary_judgments_have_no_independence_p_value(self):
        result=paired([row(),row()],'q4-fused')
        self.assertIsNone(result['mcnemar_exact_p'])

    def test_pooled_ordinal_labels_do_not_merge_unrelated_tasks(self):
        a=row();b=row('second');a.update(family='tone',question='tone');b.update(family='sentiment',question='sentiment')
        result=metric([a,b],'fp16-scan')
        self.assertEqual(result['accuracy'],1)
        self.assertIsNone(result['macro_f1']);self.assertIsNone(result['confusion'])

    def test_bootstrap_does_not_treat_auxiliary_questions_as_independent(self):
        with self.assertRaises(AssertionError):paired([row(),row()],'q4-fused',bootstrap=True)

    def test_all_matrix_modes_require_matching_quantization_metadata(self):
        modes=['fp16-scan','int8-all','q4-all']
        corpus={'seed':1,'sources':{},'cases':[{'id':'tone-test-1','family':'controlled-tone','source':{},
            'request':{'state':'Example input','questions':{'tone':{'type':'score'}}},'expected':{'tone':{'label':'1','ordinal':1}}}]}
        raw=json.dumps(corpus).encode();digest=hashlib.sha256(raw).hexdigest()
        corpus_path=SimpleNamespace(read_bytes=lambda:raw)
        report={'completed':True,'purpose':'quality-evaluation','corpus_sha256':digest,'wasm_sha256':'build',
            'metadata':{'corpus_sha256':digest,'revision':'checkpoint'},'runs':[]}
        for mode,bits in zip(modes,[0,8,4]):
            report['runs'].append({'mode':mode,'fixture':'tone-test-1','warmup':False,'round':0,
                'layers':[{'pack_ms':0,'read_ms':0} for _ in range(32)],'timing_scope':'warm-request',
                'decoder_weight_bytes_read':0,'response':{'usage':{'sequences':1,'input_tokens':10},
                'answers':{'tone':score([.1,.8,.1])}},'diagnostics':{'logits':[{'values':[[0.,1.,0.]]}]},
                'adapter':{'auxiliary_quantization':{'weight_bits':bits,'performance_eligible':False,
                    'tensors':['language_model.embed_tokens.weight','readout.weight']}}})
        path=SimpleNamespace(read_text=lambda:json.dumps(report))
        result=analyze(corpus_path,[path],modes=modes)
        self.assertEqual(result['modes'],modes)
        self.assertEqual(result['primary_metrics']['int8-all']['correct'],1)
        self.assertIn('Q8 all matrices',render(result,corpus))
        bad=copy.deepcopy(report);bad['runs'][1]['adapter']['auxiliary_quantization']['weight_bits']=0
        with self.assertRaises(AssertionError):
            analyze(corpus_path,[SimpleNamespace(read_text=lambda:json.dumps(bad))],modes=modes)


if __name__=='__main__':unittest.main()

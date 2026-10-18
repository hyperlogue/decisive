import json
from pathlib import Path
import unittest
from unittest.mock import patch

from analyze_checkpoint import ROOT, analyze


def report():
    modes=['int8','int8-tuned']
    data={'experiment':'pplx-checkpoint-phase-v1','completed':True,'timing_scope':'warm-request',
          'configuration':{'modes':modes,'fixtures':['request'],'warmups':1,'rounds':3},
          'metadata':{'revision':'test'},'startup':[],'runs':[]}
    for mode in modes:
        data['startup'].append({'mode':mode,'weight_residency':'all-decoder-layers',
                               'elapsed_ms':10000,'bytes_read':7140099072,'resident_decoder_weight_bytes':3570000000})
        for i in range(-1,3):
            elapsed=50 if mode=='int8' else 40
            data['runs'].append({'mode':mode,'fixture':'request','round':i,'warmup':i<0,
                'configuration':{'resident':True},'timing_scope':'warm-request','elapsed_ms':elapsed,'request_ms':elapsed,
                'decoder_weight_bytes_read':0,'bytes_read':0,'adapter':{'weight_residency':'all-decoder-layers'},
                'response':{'usage':{'sequences':1,'input_tokens':117}},
                'diagnostics':{'logits':[{'keys':['choice'],'values':[[1,2]]}]},
                'layers':[{'compute_ms':1,'pack_ms':0,'read_ms':0} for _ in range(32)]})
    return data


class RequestTiming(unittest.TestCase):
    def analyze(self,data):
        with patch.object(Path, 'read_text', return_value=json.dumps(data)):
            return analyze(ROOT/'benchmark-runs'/'test-report.json')

    def test_startup_is_not_part_of_request_latency(self):
        data=report();result=self.analyze(data)
        self.assertTrue(result['cases'][0]['clears_complexity_threshold'])
        data['startup'][1]['elapsed_ms']=1e9
        self.assertEqual(self.analyze(data)['cases'],result['cases'])

    def test_repeated_preparation_cannot_be_excluded(self):
        for field in ['pack_ms','read_ms']:
            data=report();data['runs'][0]['layers'][0][field]=1
            with self.assertRaises(AssertionError):self.analyze(data)
        data=report();data['runs'][0]['decoder_weight_bytes_read']=4
        with self.assertRaises(AssertionError):self.analyze(data)
        data=report();data['runs'][0]['request_ms']-=1
        with self.assertRaises(AssertionError):self.analyze(data)

    def test_single_model_reports_latency_without_a_speedup_claim(self):
        data=report();data['configuration']['modes']=['int8']
        data['runs']=[r for r in data['runs'] if r['mode']=='int8'];data['startup']=data['startup'][:1]
        result=self.analyze(data)
        self.assertEqual(result['verified_runs'],4)
        self.assertNotIn('speedup',result['cases'][0])
        self.assertFalse(result['cases'][0]['clears_complexity_threshold'])

    def test_historical_streaming_cannot_pass_warm_request_gate(self):
        data=report();data.pop('timing_scope')
        for run in data['runs']:run['bytes_read']=7140099072
        result=self.analyze(data)
        self.assertEqual(result['timing_scope'],'streamed-request-diagnostic')
        self.assertFalse(result['confirmation'])
        self.assertFalse(result['cases'][0]['clears_complexity_threshold'])

    def test_same_precision_corruption_rejected(self):
        data=report();data['runs'][-1]['diagnostics']['logits'][0]['values'][0][0]+=1
        with self.assertRaisesRegex(AssertionError,'logits'):self.analyze(data)


if __name__=='__main__':unittest.main()

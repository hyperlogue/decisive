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

    def test_queued_timing_is_not_reported_as_zero_compute(self):
        data=report();data['timing_scope']='streamed-request-diagnostic'
        for run in data['runs']:
            run.update(queued_execution=True,max_in_flight_layers=2,bytes_read=7140099072,
                       weight_buffers={'peak_in_flight_layers':2})
            for layer in run['layers']:
                layer.update(compute_ms=None,pack_ms=None,encode_enqueue_ms=.2,queue_wait_ms=.3)
        variants=self.analyze(data)['cases'][0]['variants']
        for variant in variants.values():
            self.assertIsNone(variant['median_compute_ms'])
            self.assertIsNone(variant['median_weight_preparation_ms'])
            self.assertAlmostEqual(variant['median_encode_enqueue_ms'],6.4)
            self.assertAlmostEqual(variant['median_queue_wait_ms'],9.6)
        layer=data['runs'][0]['layers'][0]
        layer['compute_ms']=0
        with self.assertRaises(AssertionError):self.analyze(data)
        layer['compute_ms']=None;layer['queue_wait_ms']=1000
        with self.assertRaises(AssertionError):self.analyze(data)

    def test_packed_grouped_q4_requires_complete_weight_reads(self):
        data=report();modes={'int8':'q4-e4-serial','int8-tuned':'q4-e4'}
        data['timing_scope']='prepacked-streamed-request';data['configuration'].update(modes=list(modes.values()),resident=False)
        for startup in data['startup']:
            startup.update(mode=modes[startup['mode']],prepacked_decoder_bytes=2009531392)
        for run in data['runs']:
            run.update(mode=modes[run['mode']],timing_scope=data['timing_scope'],decoder_weight_bytes_read=2009531392)
            run['configuration']['resident']=False
            run['adapter']['quantization']={'scheme':'q4-w4a8-e4-r4-g64'}
        self.assertTrue(self.analyze(data)['confirmation'])
        data['runs'][0]['decoder_weight_bytes_read']-=4
        with self.assertRaises(AssertionError):self.analyze(data)


if __name__=='__main__':unittest.main()

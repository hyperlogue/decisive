import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
SEED=20261018
TWEET_REV='4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66'
BANK_REV='57ec275d8078af65b7731c2a98be812d844a6d6b'


def prepare(output,cache):
    rng=random.Random(SEED);sources=[];cases=[]
    def fetch(repo,rev,path):
        url=f'https://raw.githubusercontent.com/{repo}/{rev}/{path}'
        target=cache/(repo.replace('/','-')+'-'+rev)/path
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            with urllib.request.urlopen(url,timeout=60) as response:data=response.read()
            target.write_bytes(data)
        data=target.read_bytes();sources.append({'url':url,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
        return data.decode('utf-8')
    def add(id,family,state,questions,expected,source):
        cases.append({'id':id,'family':family,'request':{'state':state,'questions':questions},'expected':expected,'source':source})
    def choice(instructions,criteria):return {'type':'choice','instructions':instructions,'criteria':criteria}
    def binary(instructions):return {'type':'noul','instructions':instructions}
    tone={'type':'score','instructions':"Rate the writer's expressed emotional tone, not how serious the situation is.",'criteria':['calm and composed','concerned or worried','angry or furious']}
    for task,per_label in [('emotion',12),('sentiment',16)]:
        base=f'datasets/{task}'
        texts=fetch('cardiffnlp/tweeteval',TWEET_REV,base+'/test_text.txt').splitlines()
        labels=fetch('cardiffnlp/tweeteval',TWEET_REV,base+'/test_labels.txt').splitlines()
        mapping=dict(line.split('\t',1) for line in fetch('cardiffnlp/tweeteval',TWEET_REV,base+'/mapping.txt').splitlines())
        assert len(texts)==len(labels)
        for label,name in mapping.items():
            selected=rng.sample([i for i,v in enumerate(labels) if v==label],per_label)
            for index in selected:
                if task=='emotion':
                    questions={'emotion':choice('Which emotion is dominant in this message?',{v:v for v in mapping.values()}),
                               'anger':binary('Is anger the dominant emotion in this message?')}
                    expected={'emotion':{'label':name},'anger':{'label':str(name=='anger').lower()}}
                else:
                    questions={'sentiment':{'type':'score','instructions':'Classify the overall sentiment expressed by this message.','criteria':[mapping[str(i)] for i in range(3)]}}
                    expected={'sentiment':{'label':label,'ordinal':int(label)}}
                add(f'{task}-{index:05}',task,texts[index],questions,expected,{'dataset':'TweetEval','split':'test','row_zero_based':index,'gold':name})
    bank=list(csv.DictReader(io.StringIO(fetch('PolyAI-LDN/task-specific-datasets',BANK_REV,'banking_data/test.csv'))))
    intents={
        'card_arrival':'A new bank card has not arrived or a question about its delivery.',
        'lost_or_stolen_card':'A bank card is lost or stolen.',
        'cash_withdrawal_charge':'A fee was charged for withdrawing cash.',
        'cash_withdrawal_not_recognised':'An unfamiliar cash withdrawal appears in the account.',
        'extra_charge_on_statement':'An additional card payment fee appears on the statement.',
        'card_payment_not_recognised':'An unfamiliar card payment appears in the account.',
        'beneficiary_not_allowed':'A bank transfer recipient is not allowed.',
        'cancel_transfer':'The user wants to cancel a bank transfer.'}
    for intent in intents:
        candidates=[i for i,row in enumerate(bank) if row['category']==intent]
        assert len(candidates)>=4,(intent,len(candidates))
        for index in rng.sample(candidates,4):
            add(f'bank-{index:04}','banking',bank[index]['text'],{'intent':choice('Which intent best matches this banking request?',intents)},
                {'intent':{'label':intent}},{'dataset':'BANKING77','split':'test','row_zero_based':index,'gold':intent,'restricted_intents':8})
    triples=[
        ('double-charge','I noticed two charges for the same purchase. Please check and refund the duplicate. Thank you.',
         'I am worried about being charged twice because I need that money for rent. Could you please investigate soon?',
         'You charged me TWICE! This is outrageous. I am furious and demand my money back immediately!'),
        ('shipping','Could you please provide an update on my delayed package? There is no rush; thank you.',
         'My package is late and I am worried it will miss the birthday. Can you check what happened?',
         'This delivery delay is ridiculous! I am fed up with your excuses. Fix this now!'),
        ('login','I cannot log in. Please send the password reset instructions when you can. Thanks.',
         'I cannot log in and I am concerned I will miss my deadline. Could you help me soon?',
         'Your login system is a complete disgrace! I am furious that I am locked out again!'),
        ('slow-app','The app is loading slowly. Could you take a look? I can wait; thanks for your help.',
         'The app is very slow and I am worried my work will not save. Can someone check it?',
         'This app is unbearably slow! I am sick of this garbage wasting my time!'),
        ('renewal','My subscription renewed. Please explain the renewal date and cancellation options. Thank you.',
         'My subscription renewed unexpectedly and I am worried about the charge. Can you help?',
         'You renewed my subscription without warning! I am furious. This is absolutely unacceptable!'),
        ('refund','Could you confirm the status of my refund? I understand processing takes time. Thanks.',
         'My refund is still pending and I am getting worried that something went wrong. Please check.',
         'Where is my refund? I am fed up with waiting and furious about your broken promises!'),
        ('booking','I see my booking was canceled. Please explain the available alternatives. Thank you.',
         'My booking was canceled and I am anxious about finding another place. Can you help soon?',
         'Canceling my booking at the last minute is outrageous! I am absolutely furious with you!'),
        ('export','The export did not finish. Please let me know the recommended steps to try again. Thanks.',
         'The export failed and I am worried about losing important records. Can you help me recover them?',
         'Your export tool failed AGAIN! I am furious about this incompetent service!')]
    for key,*texts in triples:
        for label,state in enumerate(texts):
            add(f'tone-{key}-{label}','controlled-tone',state,{'tone':tone,'anger':binary('Does the writer express anger?')},
                {'tone':{'label':str(label),'ordinal':label},'anger':{'label':str(label==2).lower()}},
                {'dataset':'authored-controlled','gold_basis':'Explicitly differentiated calm, worry and anger; not independently human-annotated.'})
    extras=[
        ('negation-no','I do not want a refund. I only need a copy of the receipt for the duplicate charge.',False,None),
        ('negation-yes','I do not want store credit; please refund the duplicate charge to my card.',True,None),
        ('past-refund','Last year I asked for a refund. Today I only need a receipt, and I am not requesting any money back.',False,None),
        ('french-angry','Ce double prélèvement est scandaleux ! Je suis furieux. Remboursez-moi immédiatement !',True,2),
        ('chinese-calm','款项没有问题，我不需要退款，只想要一张发票。谢谢您的帮助。',False,0),
        ('arabic-worried','أخشى أن يكون الخصم المزدوج خطأ، وأنا قلق بشأن المبلغ. أرجو التحقق وإعادة المبلغ الزائد.',True,1)]
    for key,state,refund,emotion in extras:
        questions={'refund':binary('Is the writer currently requesting a refund?')};expected={'refund':{'label':str(refund).lower()}}
        if emotion is not None:questions['tone']=tone;expected['tone']={'label':str(emotion),'ordinal':emotion}
        add('stress-'+key,'stress',state,questions,expected,{'dataset':'authored-controlled','gold_basis':'Authored explicit request/negation and multilingual emotional wording.'})
    for fixture in ['request','multilingual']:
        request=json.loads((ROOT/'tests/fixtures/pplx_flash'/f'{fixture}.json').read_text())
        expected={'team':{'label':'support'}} if fixture=='request' else {'department':{'label':'billing'},'refund':{'label':'true'}}
        add('regression-'+fixture,'regression',request['state'],request['questions'],expected,
            {'dataset':'existing-project-fixture','unlabeled_note':'Original multilingual tone has no authoritative gold label; report drift only.'})
    assert len(cases)==160 and len({c['id'] for c in cases})==160
    rng.shuffle(cases)
    corpus={'schema_version':1,'seed':SEED,'selection':'Stratified random sampling of fixed public test splits; labels and prompts frozen before inference. Not a full benchmark or a production sample.',
            'banking_scope':'Closed-set eight-intent subset; not full BANKING77 accuracy.',
            'sources':sources,'cases':cases}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(corpus,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({'cases':len(cases),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'families':{f:sum(c['family']==f for c in cases) for f in sorted({c['family'] for c in cases})}}))

    return corpus


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache',type=Path,default=ROOT/'.cache/pplx-quality')
    parser.add_argument('--plan',type=Path,help='Also write a 12-phase FP16/Q8/Q4 evaluation plan')
    args=parser.parse_args();corpus=prepare(args.output,args.cache)
    if args.plan:
        plan=[]
        for batch in range(4):
            ids=[c['id'] for c in corpus['cases'][batch*40:(batch+1)*40]]
            for mode in ['fp16-scan','int8-tuned','q4-fused']:
                plan.append({'id':f'quality-{batch+1}-{mode}','modes':[mode],'fixtures':ids,'corpus':True,'warmups':0,'rounds':1,'q4_check':batch==0 and mode=='q4-fused'})
        args.plan.parent.mkdir(parents=True,exist_ok=True)
        with args.plan.open('x') as stream:json.dump(plan,stream,indent=2);stream.write('\n')

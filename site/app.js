import {icon} from './lib/icons.js';
import {categories, buildRequest, readScores, welcome} from './lib/privacy.js';
import {LatestAnalysis} from './lib/latest-analysis.js';
import {probeCapabilities} from './lib/capabilities.js';
import {WorkerClient} from './lib/worker-client.js';
import {makeCard} from './lib/share.js';
import {selectModel, emptyModelState, validateModelState} from './lib/model-catalog.js';
import {observeAnalysisMotion} from './lib/analysis-motion.js';

const $ = id => document.getElementById(id);
const strings = await fetch(new URL('./lib/locales.json', import.meta.url)).then(r=>r.json());
const catalog=await fetch(new URL('./models.json',import.meta.url)).then(r=>r.json());
let manifest=selectModel(catalog);
let lang='en', ready=false, pending=false, result=null, operation=null, analysisError=null, modelError=null;
let progress=null, selected=null, shareBlob=null, shareUrl=null, signalIds=new Set(), exporting=false, loadEpoch=0;
let capabilities=null, estimate=null, compatibleWorker=false;
const states=Object.fromEntries(catalog.models.map(model=>[model.id,emptyModelState(model)]));
let statusKey='', storeState=states[manifest.id], deleteTarget=manifest.id;
const setState=(id,state)=>{states[id]=state;if(id===manifest.id)storeState=state;};
const t=(key,values={})=>(strings[lang][key]??strings.en[key]??key).replace(/\{(\w+)\}/g,(_,name)=>values[name]??'');
const bytes=n=>new Intl.NumberFormat(lang,{maximumFractionDigits:2}).format(n/1e9)+' GB';
const fileBytes=n=>n>=1e9?bytes(n):n<1000?
  new Intl.NumberFormat(lang,{style:'unit',unit:'byte',unitDisplay:'short'}).format(n):
  new Intl.NumberFormat(lang,{maximumFractionDigits:1}).format(n>=1e6?n/1e6:n/1e3)+(n>=1e6?' MB':' KB');
const errorKey=e=>strings.en[e?.message]?e.message:'storageFailed';
const shortOnStorage=()=>!storeState.ready&&estimate?.quota&&estimate.usage!=null&&
  estimate.quota-estimate.usage<storeState.remaining+16*1024*1024;
const input=$('message-input'),intro=$('intro-dialog'),setup=$('setup-dialog'),shareDialog=$('share-dialog');
const precisionReady=()=>!manifest.requiredFeatures?.includes('shader-f16')||capabilities?.shaderF16;
const canRun=()=>!!(compatibleWorker&&manifest.available&&capabilities?.gpu&&precisionReady()&&capabilities?.simd&&capabilities?.storage&&capabilities?.secure);
const setupIssue=()=>!manifest.available?'modelComingSoon':!capabilities?null:!capabilities.gpu?'webgpuUnavailable':
  !precisionReady()?'gpuPrecisionRequired':
  !capabilities.simd||!capabilities.secure?'runtimeUnsupported':!capabilities.storage?'storageUnsupported':
  shortOnStorage()?'insufficientSpace':null;
const scoreRows=new Map();
document.querySelectorAll('[data-icon]').forEach(el=>el.innerHTML=icon(el.dataset.icon));
for(const c of categories){
  const row=document.createElement('div');row.id=`score-${c.id}`;row.className='score-row';
  row.innerHTML=`<span class="score-name">${icon(c.icon)}<span class="category-label"></span></span><span class="score-bar" aria-hidden="true"><span></span></span><span class="score-value">—</span>`;
  $('score-list').append(row);scoreRows.set(c.id,row);
}
const modelOptions=[];
for(const model of catalog.models){
  const card=document.createElement('div');card.className='model-option';card.dataset.model=model.id;
  card.innerHTML=`<button type="button" class="model-pick" role="radio" aria-labelledby="model-name-${model.id}"></button><span class="model-radio" aria-hidden="true"></span><span class="model-option-text"><strong class="model-option-name" id="model-name-${model.id}"></strong><span class="model-option-meta"><span class="model-option-size"></span><a class="model-link" target="_blank" rel="noopener noreferrer">Hugging Face ↗</a></span></span><span class="model-option-actions"><span class="model-badge" aria-describedby="model-tip-${model.id}"></span><span class="model-tip" id="model-tip-${model.id}" role="tooltip"></span><button type="button" class="icon-button model-delete">${icon('trash')}</button><button type="button" class="secondary-button model-download">${icon('download')}<span></span></button></span><progress max="1" value="0" hidden></progress>`;
  card.querySelector('.model-pick').addEventListener('click',()=>changeModel(model.id));
  card.querySelector('.model-delete').addEventListener('click',()=>confirmDelete(model.id));
  const link=card.querySelector('.model-link');
  if(model.repository)link.href=`https://huggingface.co/${model.repository}/tree/${model.revision}`;else link.hidden=true;
  card.querySelector('.model-download').addEventListener('click',async()=>{
    if(operation==='download'&&model.id===manifest.id){await client.call('pause');return;}
    await changeModel(model.id);download();
  });
  $('setup-models').append(card);modelOptions.push({card,model});
}

const client=new WorkerClient(event=>{
  if(event.event==='fatal'){
    ready=false;operation=null;scheduler.setEnabled(false);analysisError={message:'workerFailed'};modelError=analysisError;paint();paintStorage();return;
  }
  progress=event;
  if(operation==='download' && event.name){
    const file=storeState.files.find(f=>f.name===event.name);
    if(file){
      if(event.phase!=='verifying')file.bytes=event.bytes;
      if(event.phase==='verified'){file.verified=true;file.state='verified';}
      else file.state=event.phase==='verifying'?'unverified':'partial';
    }
    storeState.stored=storeState.files.reduce((n,f)=>n+f.bytes,0);
    storeState.verified=storeState.files.reduce((n,f)=>n+(f.verified?f.bytes:0),0);
    storeState.remaining=storeState.files.reduce((n,f)=>n+Math.max(0,f.expected-f.bytes),0);
  }
  paintStatus();paintStorage();
});
const scheduler=new LatestAnalysis({
  async run(draft){
    operation='inference';progress=null;paintStatus();paintStorage();
    const model=manifest,request=buildRequest(draft,model.engine);
    try {
      const output=await client.call('predict',{request,modelId:model.id});
      return {request,output,modelName:model.name,scores:readScores(output.response,model.engine),elapsed:output.elapsed_ms,
        context:[{role:'assistant',content:welcome},{role:'user',content:draft}]};
    } finally {
      operation=null;progress=null;
      paintStatus();paintStorage();
    }
  },
  publish(value){result=value;pending=false;analysisError=null;paint();},
  fail(error){ready=false;pending=false;analysisError=error;scheduler.setEnabled(false);paint();},
  pending(value){pending=value;result=null;selected=null;paint();},
});

function paintLanguage(){
  document.documentElement.lang=lang;document.documentElement.dir=lang==='ar'?'rtl':'ltr';document.title=t('pageTitle');
  $('language-picker').value=lang;
  document.querySelectorAll('[data-i18n]').forEach(el=>el.textContent=t(el.dataset.i18n));
  for(const [attribute,target] of [['aria','aria-label'],['title','title'],['alt','alt']]){
    document.querySelectorAll(`[data-${attribute}]`).forEach(el=>el.setAttribute(target,t(el.dataset[attribute])));
  }
  document.querySelectorAll('[data-title]').forEach(el=>el.setAttribute('aria-label',t(el.dataset.title)));
  input.placeholder=t('placeholder');
  categories.forEach(c=>scoreRows.get(c.id).querySelector('.category-label').textContent=t(c.id));
  $('details-label').textContent=t($('analysis-details').open?'hideDetails':'showDetails');
  paint();paintStorage();paintSetup();
}
function paintStatus(){
  const working=operation==='inference'||operation==='loading';
  let label=t(!ready?'notStarted':pending?'latestWaiting':'watching');
  if(working){
    const layer=/Layer (\d+)\/32/.exec(progress?.message??'');
    label=layer?t('layer',{n:layer[1]}):t(operation==='loading'?'loading':'analyzing');
  }
  $('composer-status').textContent=label;
  $('analysis-status').lastElementChild.textContent=working?label:t(!ready?'notStarted':pending?'checking':'live');
  $('live-analysis').classList.toggle('waiting',pending||working);
  $('live-analysis').classList.toggle('inactive',!ready);
  $('composer-dot').classList.toggle('idle',!ready);
  $('composer-dot').classList.toggle('busy',ready&&(pending||working));
}
function paint(){
  const usable=ready&&!pending&&!!result;
  const active=usable?result.scores.filter(c=>c.detected):[];
  paintStatus();
  $('share-result').disabled=!usable||exporting;
  $('character-count').textContent=new Intl.NumberFormat(lang).format([...input.value].length)+' / 4,000';
  $('analysis-error').hidden=!analysisError;$('analysis-error').textContent=analysisError?t(errorKey(analysisError)):'';
  $('signal-list').hidden=!active.length;$('empty-signals').hidden=!!active.length;
  const empty=!ready?(capabilities&&!capabilities.gpu?'webgpuRequired':'needSetup'):pending?'latestWaiting':!input.value.trim()?'startTyping':'none';
  $('empty-signal-text').textContent=t(empty);
  $('analysis-caption').textContent=active.length?t('count',{n:active.length}):t(empty);
  const focus=document.activeElement?.closest('.signal-card')?.id;
  $('signal-list').replaceChildren();
  let entering=0;
  for(const c of active){
    const button=document.createElement('button');button.type='button';button.id=`signal-${c.id}`;button.className=`signal-card ${c.severity}${selected===c.id?' selected':''}`;
    if(signalIds.has(c.id))button.style.animation='none';else button.style.setProperty('--enter',entering++);
    button.innerHTML=`<span class="signal-icon">${icon(c.icon)}</span><span class="signal-text"><span class="signal-label"></span><span class="signal-meta"></span></span>`;
    button.querySelector('.signal-label').textContent=strings[lang][`short_${c.id}`]??t(c.id);
    button.querySelector('.signal-meta').textContent=t(c.severity==='strong'?'sensitive':'mild');
    button.title=`${t(c.id)} · ${t(c.severity==='strong'?'sensitive':'mild')} · ${t('modelScore')} ${c.score.toFixed(3)}`;
    button.setAttribute('aria-label',button.title);button.setAttribute('aria-pressed',String(selected===c.id));
    button.addEventListener('click',()=>{selected=c.id;$('analysis-details').open=true;paint();scoreRows.get(c.id).scrollIntoView({behavior:'smooth',block:'center'});});
    $('signal-list').append(button);
  }
  if(focus)$(focus)?.focus({preventScroll:true});signalIds=new Set(active.map(c=>c.id));
  for(const c of categories){
    const row=scoreRows.get(c.id),score=usable?result.scores.find(s=>s.id===c.id):null;
    row.className=`score-row${score?.detected?' detected':''}${c.severity==='strong'?' strong':''}${selected===c.id?' selected':''}`;
    row.querySelector('.score-bar>span').style.width=`${(score?.score??0)*100}%`;
    row.querySelector('.score-value').textContent=score?score.score.toFixed(3):'—';
  }
  $('runtime-time').textContent=usable&&Number.isFinite(result.elapsed)?
    result.elapsed<1000?t('uiTime',{n:new Intl.NumberFormat(lang,{maximumFractionDigits:0}).format(result.elapsed)}):
    t('seconds',{n:new Intl.NumberFormat(lang,{maximumFractionDigits:2}).format(result.elapsed/1000)}):'—';
  const tokens=usable?result.output.response.usage?.input_tokens:null;
  $('runtime-tokens').textContent=Number.isSafeInteger(tokens)&&tokens>=0?new Intl.NumberFormat(lang).format(tokens):'—';
  $('raw-output').textContent=JSON.stringify(usable?{input:result.request,output:result.output.response,backend:result.output.backend,elapsed_ms:result.elapsed}:
    analysisError?{error:analysisError.detail??t(errorKey(analysisError))}:{status:t(empty)},null,2);
}
function paintSetup(){
  paintModelOptions();
  document.querySelectorAll('[data-i18n="downloadNote"]').forEach(el=>el.textContent=t(manifest.available?'downloadNote':'modelComingSoon'));
  const issue=modelError?errorKey(modelError):setupIssue();
  $('setup-error').hidden=!issue||issue==='modelComingSoon'||issue==='gpuPrecisionRequired';
  $('setup-error-text').textContent=issue?t(issue):'';
  $('legacy-model').hidden=!storeState.legacyStored;
  $('legacy-size').textContent=t('legacyNote',{size:fileBytes(storeState.legacyStored??0)});
  $('delete-legacy').disabled=!!operation;
  $('start-analysis').disabled=!storeState.ready||!canRun()||!!operation;
  $('start-analysis').textContent=t(manifest.available?'useModel':'comingSoon');
  $('stop-model').hidden=operation!=='inference'&&operation!=='loading';
}
function paintModelOptions(){
  for(const {card,model} of modelOptions){
    const state=states[model.id],chosen=model.id===manifest.id,pick=card.querySelector('.model-pick');
    const reason=!capabilities?null:!capabilities.gpu?'webgpuUnavailable':model.requiredFeatures?.includes('shader-f16')&&!capabilities.shaderF16?'needsF16':null;
    const blocked=model.available&&!!reason;
    const downloading=chosen&&operation==='download';
    card.classList.toggle('selected',chosen);card.classList.toggle('ready',state.ready);card.classList.toggle('blocked',blocked);
    pick.setAttribute('aria-checked',String(chosen));pick.disabled=blocked||!!operation&&!chosen;
    card.querySelector('.model-option-name').textContent=model.name;
    card.querySelector('.model-option-size').textContent=!model.available?t('sizePending'):
      state.ready?t('modelSizeStored',{size:fileBytes(state.total)}):
      state.stored?t('setupSize',{size:fileBytes(state.total),stored:fileBytes(state.stored)}):t('modelSizeDownload',{size:fileBytes(state.total)});
    const badge=card.querySelector('.model-badge');
    badge.hidden=model.available&&!state.ready&&!blocked;badge.textContent=t(!model.available?'comingSoon':blocked?'disabled':'ready');
    badge.tabIndex=blocked?0:-1;card.querySelector('.model-tip').textContent=blocked?t(reason,{model:model.name}):'';
    const bar=card.querySelector('progress');bar.hidden=!downloading;bar.max=state.total||1;bar.value=state.total-state.remaining;
    const remove=card.querySelector('.model-delete');
    remove.hidden=!state.stored||downloading;remove.disabled=!!operation;remove.title=t('deleteModel');remove.setAttribute('aria-label',`${t('deleteModel')} · ${model.name}`);
    const get=card.querySelector('.model-download'),label=t(downloading?'pause':state.stored?'resume':'downloadShort');
    get.hidden=!model.available||state.ready||blocked;
    get.disabled=!downloading&&(!!operation||blocked||!compatibleWorker||!capabilities?.gpu||!capabilities.simd||!capabilities.storage||!capabilities.secure||chosen&&!!shortOnStorage());
    get.querySelector('span').textContent=label;get.setAttribute('aria-label',`${label} · ${model.name}`);
    card.querySelector('.model-link').title=t('modelSource');
  }
}
function paintStorage(){
  $('runtime-model').textContent=manifest.name;
  let message=statusKey?t(statusKey):'';
  if(operation==='download')message='';
  if(operation==='loading')message=t('loading');
  $('setup-status').textContent=message;
  paintSetup();
}
async function refreshStorage(){
  const selected=manifest;
  try{
    if(!operation){
      const state=await client.call('inspect',{modelId:selected.id});
      if(selected!==manifest)return;
      setState(selected.id,validateModelState(selected,state));compatibleWorker=true;
      if(modelError?.message==='versionMismatch')modelError=null;
      for(const model of catalog.models){
        if(model===selected||!model.available)continue;
        try{states[model.id]=validateModelState(model,await client.call('inspect',{modelId:model.id}));}catch{}
      }
    }
    estimate=await navigator.storage?.estimate?.();
  }catch(error){if(selected!==manifest)return;modelError=error;if(error.message==='versionMismatch')compatibleWorker=false;}
  paintStorage();
}
function openSetup(){
  intro.close();paintSetup();
  if(!setup.open)setup.showModal();
  refreshStorage();
}
function openLegacyModels(){
  if(location.hash!=='#models')return false;
  history.replaceState(null,'',location.pathname+location.search);
  openSetup();
  return true;
}
async function download(){
  if(operation||!canRun()||shortOnStorage())return;
  ready=false;scheduler.setEnabled(false);operation='download';modelError=null;statusKey='';progress=null;paint();paintStorage();
  navigator.storage?.persist?.().catch(()=>{});
  try{setState(manifest.id,validateModelState(manifest,await client.call('download',{modelId:manifest.id})));statusKey='';}
  catch(error){if(error.message==='paused')statusKey='paused';else modelError=error;}
  finally{operation=null;progress=null;await refreshStorage();paint();}
}
async function load(){
  if(operation||!canRun())return;
  const epoch=++loadEpoch;
  setup.close();ready=false;scheduler.setEnabled(false);operation='loading';progress=null;analysisError=null;paint();paintStorage();
  try{
    await client.call('load',{modelId:manifest.id});
    if(epoch===loadEpoch){ready=true;scheduler.setEnabled(true);}
    else await client.call('unload');
  }catch(error){analysisError=error;ready=false;}
  finally{operation=null;progress=null;paint();paintStorage();}
}
async function stop(){
  loadEpoch++;ready=false;scheduler.setEnabled(false);pending=false;
  await client.call('pause');paint();
}
async function changeModel(id){
  if(operation||id===manifest.id)return;
  ready=false;compatibleWorker=false;scheduler.setEnabled(false);operation='switching';paintStorage();
  try{await client.call('unload');manifest=selectModel(catalog,id);storeState=states[id];analysisError=null;modelError=null;statusKey='';}
  catch(error){modelError=error;}
  finally{operation=null;await refreshStorage();paint();}
}
input.addEventListener('input',()=>{analysisError=null;scheduler.update(input.value);});
$('language-picker').addEventListener('change',e=>{lang=e.target.value;paintLanguage();});
$('analysis-details').addEventListener('toggle',()=>{$('details-label').textContent=t($('analysis-details').open?'hideDetails':'showDetails');});
$('open-setup').addEventListener('click',openSetup);
$('close-setup').addEventListener('click',()=>setup.close());
$('close-intro').addEventListener('click',()=>intro.close());
$('intro-continue').addEventListener('click',openSetup);
$('start-analysis').addEventListener('click',load);
$('stop-model').addEventListener('click',stop);
function confirmDelete(id){deleteTarget=id;$('delete-description').textContent=t('deleteDescription',{size:fileBytes(states[id].stored)});$('delete-dialog').showModal();}
$('delete-legacy').addEventListener('click',()=>{deleteTarget='legacy';$('delete-description').textContent=t('deleteLegacyDescription',{size:fileBytes(storeState.legacyStored)});$('delete-dialog').showModal();});
$('cancel-delete').addEventListener('click',()=>$('delete-dialog').close());
$('confirm-delete').addEventListener('click',async()=>{
  $('delete-dialog').close();if(operation)return;
  ready=false;scheduler.setEnabled(false);operation='delete';modelError=null;paint();paintStorage();
  try{
    if(deleteTarget==='legacy')await client.call('removeLegacy');
    else setState(deleteTarget,validateModelState(selectModel(catalog,deleteTarget),await client.call('remove',{modelId:deleteTarget})));
    statusKey='deleted';
  }catch(error){modelError=error;}
  finally{operation=null;await refreshStorage();paint();}
});
window.addEventListener('hashchange',openLegacyModels);
const channel=typeof BroadcastChannel==='function'?new BroadcastChannel('decisive-privacy-storage'):null;
channel?.addEventListener('message',async()=>{
  if(!operation){await refreshStorage();if(!storeState.ready){ready=false;scheduler.setEnabled(false);paint();}}
});
window.addEventListener('focus',()=>{if(setup.open&&!operation)refreshStorage();});

const privacy=$('privacy-help'),tooltip=$('privacy-tooltip');let privacyPinned=false;
function togglePrivacy(open){tooltip.hidden=!open;privacy.setAttribute('aria-expanded',String(open));}
privacy.addEventListener('click',()=>{privacyPinned=!privacyPinned;togglePrivacy(privacyPinned);});
privacy.addEventListener('mouseenter',()=>togglePrivacy(true));privacy.addEventListener('focus',()=>togglePrivacy(true));
privacy.closest('.privacy-wrap').addEventListener('mouseleave',()=>{if(!privacyPinned)togglePrivacy(false);});
privacy.addEventListener('blur',()=>{if(!privacyPinned)togglePrivacy(false);});
document.addEventListener('keydown',e=>{if(e.key==='Escape'){privacyPinned=false;togglePrivacy(false);}});
document.addEventListener('click',e=>{if(!e.target.closest('.privacy-wrap')){privacyPinned=false;togglePrivacy(false);}});
for(const dialog of document.querySelectorAll('dialog'))dialog.addEventListener('click',e=>{
  if(e.target!==dialog)return;const box=dialog.getBoundingClientRect();
  if(e.clientX<box.left||e.clientX>box.right||e.clientY<box.top||e.clientY>box.bottom)dialog.close();
});
$('close-share').addEventListener('click',()=>shareDialog.close());
$('share-result').addEventListener('click',async()=>{
  if(!result||pending||!ready)return;
  const snapshot=structuredClone(result),locale=lang;
  const translate=(key,values={})=>(strings[locale][key]??strings.en[key]??key).replace(/\{(\w+)\}/g,(_,name)=>values[name]??'');
  exporting=true;paint();$('export-status').textContent='';
  try{
    shareBlob=await makeCard(snapshot,locale,translate);if(shareUrl)URL.revokeObjectURL(shareUrl);
    shareUrl=URL.createObjectURL(shareBlob);$('share-image').src=shareUrl;
    const file=new File([shareBlob],'privacy-analysis.png',{type:'image/png'});
    let canShare=false;try{canShare=!!navigator.share&&!!navigator.canShare?.({files:[file]});}catch{}
    $('native-share').hidden=!canShare;$('share-feedback').textContent=t('shareReminder');shareDialog.showModal();
  }catch{$('export-status').textContent=t('renderFailed');}
  finally{exporting=false;paint();}
});
$('download-image').addEventListener('click',()=>{
  if(!shareUrl)return;const a=document.createElement('a');a.href=shareUrl;a.download='privacy-analysis.png';document.body.append(a);a.click();a.remove();$('share-feedback').textContent=t('downloaded');
});
$('native-share').addEventListener('click',async()=>{
  if(!shareBlob)return;try{await navigator.share({files:[new File([shareBlob],'privacy-analysis.png',{type:'image/png'})],title:t('cardTitle')});$('share-feedback').textContent=t('shared');}
  catch(error){if(error.name!=='AbortError')$('share-feedback').textContent=t('shareFailed');}
});
shareDialog.addEventListener('close',()=>{if(shareUrl)URL.revokeObjectURL(shareUrl);shareUrl=null;shareBlob=null;$('share-image').removeAttribute('src');});
function adaptViewport(){
  const height=Math.round(window.visualViewport?.height??innerHeight),wasCompact=document.documentElement.classList.contains('compact-viewport');
  document.documentElement.style.setProperty('--app-height',`${height}px`);document.documentElement.classList.toggle('compact-viewport',height<560);
  if(height<560&&!wasCompact&&document.activeElement===input)$('analysis-scroll').scrollTop=0;
}
window.visualViewport?.addEventListener('resize',adaptViewport);window.addEventListener('resize',adaptViewport);adaptViewport();
paintLanguage();
observeAnalysisMotion($('live-analysis'));
const checked=await probeCapabilities();
try{
  const worker=await client.call('capabilities');
  checked.storage=checked.storage&&worker.sync;checked.gpu=!!worker.gpu;checked.shaderF16=!!worker.shaderF16;
}catch{checked.storage=false;checked.gpu=false;}
capabilities=checked;paintSetup();paint();
await refreshStorage();
if(!openLegacyModels())intro.showModal();
if('serviceWorker' in navigator){
  let controlled=!!navigator.serviceWorker.controller;
  const reloadWhenIdle=()=>{if(document.hidden&&!operation&&!input.value)location.reload();};
  const release=/\/releases\/(\w+)\//.exec(import.meta.url)?.[1];
  navigator.serviceWorker.addEventListener('controllerchange',async()=>{
    if(!controlled){controlled=true;return;}
    try{
      const latest=(await fetch(new URL('./release.json',document.baseURI),{cache:'no-store'}).then(r=>r.json())).version;
      if(!release||latest===release)return;
    }catch{return;}
    const banner=$('update-banner');
    if(!banner.open){if(document.querySelector('dialog:modal'))banner.showModal();else banner.show();}
    reloadWhenIdle();
  });
  navigator.serviceWorker.register(new URL('./sw.js',document.baseURI),{scope:'./',updateViaCache:'none'}).then(registration=>{
    const check=()=>registration.update().catch(()=>{});
    document.addEventListener('visibilitychange',()=>{if(!document.hidden)check();else if($('update-banner').open)reloadWhenIdle();});
    setInterval(check,30*60*1000);
  }).catch(()=>{});
}
$('reload-update').addEventListener('click',()=>location.reload());
window.addEventListener('pagehide',()=>{if(shareUrl)URL.revokeObjectURL(shareUrl);});

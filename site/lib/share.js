import {categories} from './privacy.js';
import {icon} from './icons.js';
function wrapText(ctx,text,maxWidth,locale){
  const lines=[];
  for(const paragraph of text.split('\n')){
    if(!paragraph){lines.push('');continue;}
    const words=typeof Intl.Segmenter==='function'?[...new Intl.Segmenter(locale,{granularity:'word'}).segment(paragraph)].map(x=>x.segment):[...paragraph];
    let line='';
    for(const word of words){
      if(ctx.measureText(line+word).width<=maxWidth){line+=word;continue;}
      if(line.trim())lines.push(line.trimEnd());line='';
      if(ctx.measureText(word).width<=maxWidth){line=word.trimStart();continue;}
      const letters=typeof Intl.Segmenter==='function'?[...new Intl.Segmenter(locale,{granularity:'grapheme'}).segment(word)].map(x=>x.segment):[...word];
      for(const letter of letters){if(ctx.measureText(line+letter).width>maxWidth&&line){lines.push(line);line='';}line+=letter;}
    }
    if(line)lines.push(line.trimEnd());
  }
  return lines;
}
export async function makeCard(snapshot, lang, t){
  await document.fonts.load('32px "Noto Sans SC"', snapshot.context.map(message=>message.content).join(''));
  await document.fonts.ready;
  const canvas=document.createElement('canvas'),ctx=canvas.getContext('2d');
  if(!ctx)throw Error('Canvas unavailable');
  const rtl=lang==='ar',font='Arial, "Noto Sans SC", sans-serif';
  ctx.font=`27px ${font}`;const contextLines=wrapText(ctx,snapshot.context[0].content,936,lang);
  ctx.font=`32px ${font}`;const messageLines=wrapText(ctx,snapshot.context[1].content,888,lang);
  const flagged=snapshot.scores.filter(s=>s.detected);
  const messageHeight=messageLines.length*48+48;
  canvas.width=1080;canvas.height=180+contextLines.length*40+64+messageHeight+84+Math.max(1,Math.ceil(flagged.length/2))*80+130;
  ctx.fillStyle='#fff';ctx.fillRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle='#4f46e5';ctx.fillRect(0,0,canvas.width,8);
  const round=(x,y,w,h,r,color)=>{ctx.fillStyle=color;ctx.beginPath();ctx.roundRect(x,y,w,h,r);ctx.fill();};
  const draw=(text,y,size=24,color='#667085',bold=false,x=null)=>{ctx.font=`${bold?'600':'400'} ${size}px ${font}`;ctx.fillStyle=color;ctx.direction=rtl?'rtl':'ltr';ctx.textAlign=rtl?'right':'left';ctx.fillText(text,x??(rtl?1008:72),y);};
  draw(t('cardTitle'),76,38,'#101828',true);
  draw(snapshot.modelName+' · '+t('modelScore'),112,20);
  ctx.fillStyle='#e4e7ec';ctx.fillRect(72,140,936,1);
  let y=180;
  draw(t('cardContext'),y,20,'#475467',true);y+=37;
  const contextRtl=/^[^\p{L}]*[\p{Script=Arabic}\p{Script=Hebrew}]/u.test(snapshot.context[0].content);
  for(const line of contextLines){
    ctx.font=`27px ${font}`;ctx.fillStyle='#344054';ctx.direction=contextRtl?'rtl':'ltr';ctx.textAlign=contextRtl?'right':'left';
    ctx.fillText(line,contextRtl?1008:72,y);y+=40;
  }y+=25;
  draw(t('cardMessage'),y,20,'#475467',true);y+=18;
  round(72,y,936,messageHeight,12,'#f2f4f7');y+=47;
  const draftRtl=/^[^\p{L}]*[\p{Script=Arabic}\p{Script=Hebrew}]/u.test(snapshot.context[1].content);
  for(const line of messageLines){ctx.font=`32px ${font}`;ctx.fillStyle='#101828';ctx.direction=draftRtl?'rtl':'ltr';ctx.textAlign=draftRtl?'right':'left';ctx.fillText(line,draftRtl?984:96,y);y+=48;}y+=38;
  draw(t('cardResults'),y,20,'#475467',true);y+=24;
  if(!flagged.length){draw(t('cardNone'),y+40,26,'#475467');y+=80;}
  else{
    flagged.forEach((s,i)=>{
      const x=72+(rtl?1-i%2:i%2)*480,top=y+Math.floor(i/2)*80;
      const strong=s.severity==='strong';round(x,top,456,66,12,strong?'#fee4e2':'#fef0c7');
      ctx.strokeStyle=strong?'#b42318':'#93370d';ctx.lineWidth=2;ctx.save();ctx.translate(x+20,top+18);ctx.scale(1.25,1.25);
      const c=categories.find(c=>c.id===s.id);const svg=new DOMParser().parseFromString(icon(c.icon),'image/svg+xml');
      for(const p of svg.querySelectorAll('path'))ctx.stroke(new Path2D(p.getAttribute('d')));
      for(const circle of svg.querySelectorAll('circle')){ctx.beginPath();ctx.arc(Number(circle.getAttribute('cx')),Number(circle.getAttribute('cy')),Number(circle.getAttribute('r')),0,Math.PI*2);ctx.stroke();}
      for(const rect of svg.querySelectorAll('rect')){ctx.beginPath();ctx.roundRect(Number(rect.getAttribute('x')),Number(rect.getAttribute('y')),Number(rect.getAttribute('width')),Number(rect.getAttribute('height')),Number(rect.getAttribute('rx')||0));ctx.stroke();}
      ctx.restore();ctx.font=`600 23px ${font}`;ctx.fillStyle=strong?'#b42318':'#93370d';ctx.direction=rtl?'rtl':'ltr';ctx.textAlign=rtl?'right':'left';ctx.fillText(t(s.id),rtl?x+330:x+66,top+40);
      ctx.direction='ltr';ctx.textAlign='right';ctx.font=`21px ${font}`;ctx.fillText(s.score.toFixed(3),x+435,top+40);
    });y+=Math.ceil(flagged.length/2)*80;
  }
  y+=27;ctx.fillStyle='#e4e7ec';ctx.fillRect(72,y,936,1);y+=36;
  draw(snapshot.modelName+' · '+t('seconds',{n:(snapshot.elapsed/1000).toFixed(1)}),y,20);
  draw('decisive.hyperlogue.tech/demo/private-info-detector',y+33,18,'#4f46e5');
  return new Promise((resolve,reject)=>canvas.toBlob(blob=>blob?resolve(blob):reject(Error('PNG export failed')),'image/png'));
}

export function observeAnalysisMotion(panel) {
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');
  const selector='.section-heading, #inference-metrics, #analysis-summary, #analysis-details, #analysis-footer, .score-value, #runtime-time, #runtime-tokens, #empty-signal-text, #raw-output';
  const animations=new Map();
  let previous=new Map(), frame;
  function capture(animate) {
    const next=new Map(), origin=panel.getBoundingClientRect();
    for(const el of panel.querySelectorAll(selector)) {
      animations.get(el)?.cancel();
      const rect=el.getBoundingClientRect(), old=previous.get(el);
      const state={top:rect.top-origin.top,text:el.textContent,visible:rect.width>0&&rect.height>0};
      next.set(el,state);
      if(!animate||reduced.matches||!state.visible)continue;
      const delta=old?.visible?old.top-state.top:0;
      const leaf=!el.querySelector(selector);
      const changed=leaf&&old?.text!==state.text;
      if(Math.abs(delta)>.5||changed) {
        const animation=el.animate([
          {transform:`translateY(${delta}px)`,opacity:changed ? 0.35 : 1},
          {transform:'translateY(0)',opacity:1},
        ],{duration:220,easing:'cubic-bezier(.2,.7,.3,1)'});
        animations.set(el,animation);
        animation.finished.then(()=>{if(animations.get(el)===animation)animations.delete(el);}).catch(()=>{});
      }
    }
    previous=next;
  }
  capture(false);
  const observer=new MutationObserver(()=>{
    cancelAnimationFrame(frame);
    frame=requestAnimationFrame(()=>capture(true));
  });
  observer.observe(panel,{subtree:true,childList:true,characterData:true,attributes:true,attributeFilter:['hidden','open','class','disabled']});
  reduced.addEventListener('change',()=>{
    if(reduced.matches)for(const animation of animations.values())animation.cancel();
  });
}

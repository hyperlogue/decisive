self.onmessage=async({data:{id,requests}})=>{
  try{
    const buffers=await Promise.all(requests.map(async({url,file,start,end})=>{
      if(file){
        const buffer=await file.slice(start,end).arrayBuffer();
        if(buffer.byteLength!==end-start)throw Error('Incomplete local layer range');
        return buffer;
      }
      const r=await fetch(url,{headers:{Range:`bytes=${start}-${end-1}`}});
      if(r.status!==206)throw Error(`Range support required: ${r.status}`);
      const buffer=await r.arrayBuffer();
      if(buffer.byteLength!==end-start)throw Error('Incomplete layer range');
      return buffer;
    }));
    self.postMessage({id,buffers},buffers);
  }catch(error){self.postMessage({id,error:String(error)});}
};

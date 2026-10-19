export async function cachePackedModel(manifest, Hasher, progress, signal) {
  if (!navigator.storage?.getDirectory) throw Error('Browser storage requires HTTPS and OPFS support.');
  const root = await navigator.storage.getDirectory();
  const dir = await root.getDirectoryHandle(`pplx-q4-e4-${manifest.revision}`, {create: true});
  let receipt;
  try { receipt = JSON.parse(await (await (await dir.getFileHandle('verified.json')).getFile()).text()); }
  catch { receipt = {}; }
  if (!receipt || typeof receipt !== 'object' || Array.isArray(receipt)) receipt = {};
  const files = {}, total = Object.values(manifest.files).reduce((n, f) => n + f.bytes, 0);
  let completed = 0, networkBytes = 0;
  for (const [name, spec] of Object.entries(manifest.files)) {
    signal.throwIfAborted();
    if (!/^[a-zA-Z0-9_.-]+$/.test(name) || !/^[a-f0-9]{64}$/.test(spec.sha256) || !Number.isSafeInteger(spec.bytes) || spec.bytes < 1) throw Error('Invalid cache manifest');
    const fileHandle = await dir.getFileHandle(name, {create: true});
    let file = await fileHandle.getFile();
    if (file.size !== spec.bytes || receipt[name]?.sha256 !== spec.sha256 || receipt[name]?.modified !== file.lastModified) {
      delete receipt[name];
      const access = await fileHandle.createSyncAccessHandle();
      let hash = new Hasher(), finished = false;
      let offset = access.getSize(), flushed = offset, lastProgress = 0;
      try {
        if (offset > spec.bytes) { access.truncate(0); offset = 0; }
        const buffer = new Uint8Array(4 * 1024 * 1024);
        for (let pos = 0; pos < offset;) {
          signal.throwIfAborted();
          const bytes = access.read(buffer.subarray(0, Math.min(buffer.length, offset - pos)), {at: pos});
          if (!bytes) throw Error(`Cannot read cached ${name}`);
          hash.update(buffer.subarray(0, bytes)); pos += bytes;
          progress({message: `Verifying ${name}`, completed: completed + pos, total});
          await new Promise(resolve => setTimeout(resolve, 0));
        }
        if (offset < spec.bytes) {
          const response = await fetch(new URL(`model/${name}`, import.meta.url), {
            headers: offset ? {Range: `bytes=${offset}-${spec.bytes - 1}`} : {}, signal, credentials: 'omit', cache: 'no-store',
          });
          if (!response.ok || !response.body) throw Error(`${name}: HTTP ${response.status}`);
          if (response.status === 200) {
            if (offset) { hash.free(); hash = new Hasher(); }
            offset = 0; access.truncate(0);
          } else if (response.status !== 206 || response.headers.get('Content-Range') !== `bytes ${offset}-${spec.bytes - 1}/${spec.bytes}`) throw Error(`Invalid resumed range: ${name}`);
          const reader = response.body.getReader();
          try {
            while (true) {
              signal.throwIfAborted();
              const {value, done} = await reader.read();
              if (done) break;
              if (offset + value.length > spec.bytes) throw Error(`Unexpected size: ${name}`);
              for (let written = 0; written < value.length;) {
                const n = access.write(value.subarray(written), {at: offset + written});
                if (!n) throw Error('Browser storage is full or unavailable');
                written += n;
              }
              hash.update(value); offset += value.length; networkBytes += value.length;
              if (offset - flushed >= 64 * 1024 * 1024) { access.flush(); flushed = offset; }
              if (performance.now() - lastProgress > 150) {
                progress({message: `Downloading ${name}`, completed: completed + offset, total});
                lastProgress = performance.now();
              }
            }
          } finally { await reader.cancel().catch(() => {}); }
        }
        const digest = hash.finish(); finished = true;
        if (offset !== spec.bytes || digest !== spec.sha256) {
          access.truncate(0);
          throw Error(`Checksum or size mismatch: ${name}. Tap Prepare to retry this file.`);
        }
      } finally {
        if (!finished) hash.free();
        try { access.flush(); } finally { access.close(); }
      }
      file = await fileHandle.getFile();
      receipt[name] = {sha256: spec.sha256, modified: file.lastModified};
      const receiptHandle = await (await dir.getFileHandle('verified.json', {create: true})).createSyncAccessHandle();
      try {
        const bytes = new TextEncoder().encode(JSON.stringify(receipt));
        receiptHandle.truncate(0);
        for (let pos = 0; pos < bytes.length;) {
          const n = receiptHandle.write(bytes.subarray(pos), {at: pos});
          if (!n) throw Error('Cannot save cache receipt');
          pos += n;
        }
        receiptHandle.flush();
      } finally { receiptHandle.close(); }
    }
    files[name] = file; completed += file.size;
    progress({message: `Cached ${name}`, completed, total});
  }
  return {files, network_bytes: networkBytes, total_bytes: total};
}

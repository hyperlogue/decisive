export function createEmbeddingCache(readRow, {rowBytes = 5120, capacityBytes = 16 * 1024 * 1024} = {}) {
  if (!Number.isSafeInteger(rowBytes) || rowBytes <= 0 || !Number.isSafeInteger(capacityBytes) || capacityBytes < 0) throw Error('Invalid embedding cache size');
  const capacity = Math.floor(capacityBytes / rowBytes), entries = new Map();
  const keep = (id, bytes) => {
    entries.delete(id);
    if (!capacity) return;
    entries.set(id, bytes);
    while (entries.size > capacity) entries.delete(entries.keys().next().value);
  };
  return {
    clear() { entries.clear(); },
    async load(ids) {
      if (!ids.every(id => Number.isSafeInteger(id) && id >= 0)) throw Error('Invalid embedding token ID');
      const unique = [...new Set(ids)], selected = new Map(), missing = [];
      for (const id of unique) {
        const bytes = entries.get(id);
        if (bytes) { selected.set(id, bytes); keep(id, bytes); }
        else missing.push(id);
      }
      for (let first = 0; first < missing.length; first += 8) {
        await Promise.all(missing.slice(first, first + 8).map(async id => {
          const bytes = await readRow(id);
          if (!(bytes instanceof Uint8Array) || bytes.byteLength !== rowBytes) throw Error('Invalid embedding row length');
          selected.set(id, bytes); keep(id, bytes);
        }));
      }
      const rows = new Uint8Array(ids.length * rowBytes);
      ids.forEach((id, index) => rows.set(selected.get(id), index * rowBytes));
      return {rows, hits: unique.length - missing.length, misses: missing.length,
        retained_bytes: entries.size * rowBytes, capacity_bytes: capacity * rowBytes};
    },
  };
}

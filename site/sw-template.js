const ASSETS = __ASSETS__;
const PREFIX = `decisive-privacy-app:${self.registration.scope}:`;
const CACHE = PREFIX + '__VERSION__';
const urls = new Set(ASSETS.map(path => new URL(path, self.registration.scope).href));
const FONTS = new URL('__FONTS__', self.registration.scope).href;
self.addEventListener('install', event => {
  event.waitUntil(caches.open(CACHE).then(cache => cache.addAll(ASSETS)));
});
self.addEventListener('activate', event => {
  event.waitUntil((async () => {
    for (const key of await caches.keys()) if (key.startsWith(PREFIX) && key !== CACHE) await caches.delete(key);
    await self.clients.claim();
  })());
});
self.addEventListener('fetch', event => {
  const request = event.request, url = new URL(request.url);
  if (request.method !== 'GET' || url.origin !== self.location.origin) return;
  if (request.mode === 'navigate' && url.pathname.startsWith(new URL(self.registration.scope).pathname)) {
    event.respondWith(caches.open(CACHE).then(cache=>cache.match('./index.html')).then(cached=>cached || fetch(request)));
  } else if (urls.has(url.href)) {
    event.respondWith(caches.open(CACHE).then(cache=>cache.match(request)).then(cached=>cached || fetch(request)));
  } else if (url.href.startsWith(FONTS)) {
    event.respondWith(caches.open(CACHE).then(async cache => (await cache.match(request)) ||
      fetch(request).then(response => { if (response.ok) cache.put(request, response.clone()); return response; })));
  }
});

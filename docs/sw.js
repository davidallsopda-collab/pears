// Lets Pears open like an app and keep showing the last numbers if the signal drops.
const CACHE = "pears-v1";
const SHELL = ["./", "index.html", "manifest.webmanifest", "icons/icon-192.png", "icons/icon-512.png", "icons/icon.svg"];
self.addEventListener("install", function (e) {
  e.waitUntil(caches.open(CACHE).then(function (c) { return c.addAll(SHELL); }).then(function () { return self.skipWaiting(); }));
});
self.addEventListener("activate", function (e) {
  e.waitUntil(caches.keys().then(function (keys) {
    return Promise.all(keys.filter(function (k) { return k !== CACHE; }).map(function (k) { return caches.delete(k); }));
  }).then(function () { return self.clients.claim(); }));
});
self.addEventListener("fetch", function (e) {
  var req = e.request;
  if (req.method !== "GET" || new URL(req.url).origin !== location.origin) return;
  var path = new URL(req.url).pathname;
  if (/\.json$|index\.html$|\/$/.test(path)) {
    e.respondWith(fetch(req).then(function (res) {
      var copy = res.clone(); caches.open(CACHE).then(function (c) { var u = new URL(req.url); u.search = ""; c.put(u.href, copy); }); return res;
    }).catch(function () { return caches.match(req, { ignoreSearch: true }); }));
  } else {
    e.respondWith(caches.match(req).then(function (hit) { return hit || fetch(req); }));
  }
});

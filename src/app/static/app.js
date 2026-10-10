// Spotify Refine shell: session, login/landing, routing, and the Home pipeline view.
// review.js (sorting views) and gallery.js (taste gallery) hook in through window.App.
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (p) => `${Math.round(100 * p)}%`;
const songs = (n) => `${n.toLocaleString()} song${n === 1 ? "" : "s"}`;

async function api(path, body) {
  const opts = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, opts);
  const j = await r.json().catch(() => ({ ok: false, error: `HTTP ${r.status}` }));
  if (j.ok === false) throw new Error(j.error || "Request failed");
  return j;
}

let toastTimer;
function toast(msg, error = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (error ? " error" : "");
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), error ? 6000 : 2500);
}
const guard = (fn) => async (...a) => { try { return await fn(...a); } catch (e) { toast(e.message, true); } };

const VIEWS = ["home", "review", "clusters", "playlists", "apply", "gallery", "listening"];
const NEEDS_LIBRARY = ["listening"];          // views that only need a pulled library, not a finished sort
const App = { session: null, view: "home", ready: false, hooks: { ready: [], view: [] } };
window.App = App;

// ---------------------------------------------------------------- session

async function refreshSession() {
  const s = await api("/api/session");
  const wasReady = App.ready;
  App.session = s;
  App.ready = s.ready;
  const showLanding = s.mode === "spotify" && !s.user;
  $("#landing").hidden = !showLanding;
  $("#shell").hidden = showLanding;
  if (showLanding) renderLanding();
  else {
    renderChrome();
    renderHome();
    if (App.ready && (!wasReady || App.needsReload)) {
      App.needsReload = false;
      for (const fn of App.hooks.ready) await fn();
    }
  }
  schedulePoll();
  return s;
}

let pollTimer;
function schedulePoll() {
  clearTimeout(pollTimer);
  if (App.session?.jobs.running) pollTimer = setTimeout(guard(pollJobs), 1000);
}
async function pollJobs() {
  await refreshSession();
  if (App.session.jobs.running) return;
  // The run just ended: reload the review data from disk and say how it went.
  const steps = Object.values(App.session.jobs.steps);
  App.needsReload = true;
  await refreshSession();
  if (steps.some((j) => j.status === "failed")) toast("A step failed. Its log is on the Home page.", true);
  else if (App.session.jobs.steps.sort.status === "done") toast("Sorted. Your suggestions are ready.");
  else toast("Done.");
}

// ---------------------------------------------------------------- landing

function renderLanding() {
  const s = App.session;
  const err = new URLSearchParams(location.search).get("error");
  $("#landing-error").hidden = !err;
  $("#landing-error").textContent = err || "";
  $("#setup-help").hidden = s.configured;
  $("#redirect-uri").textContent = s.redirect_uri;
  const login = $("#login-btn");
  login.classList.toggle("disabled", !s.configured);
  login.setAttribute("aria-disabled", String(!s.configured));
}

// ---------------------------------------------------------------- chrome (sidebar, user)

function renderChrome() {
  const s = App.session;
  $$(".nav a.needs-ready").forEach((a) => a.classList.toggle("disabled", !App.ready));
  $$(".nav a.needs-library").forEach((a) => a.classList.toggle("disabled", !s.done.pull));
  const u = s.user;
  $("#user").innerHTML = u
    ? `<span class="user-pill"><span class="avatar">${u.image ? `<img src="${esc(u.image)}" alt="">` : esc(u.name[0])}</span>${esc(u.name)}</span>
       <button class="btn small outline" id="logout">Log out</button>`
    : s.mode === "demo" ? `<a class="btn small primary" href="/login">Log in with Spotify</a>` : "";
  $("#mode-card").innerHTML = s.mode === "demo"
    ? `<strong>Demo library</strong>3,000 synthetic songs in 25 playlists. Nothing here touches Spotify.
       <button class="btn small primary" id="use-spotify">Use my Spotify</button>`
    : `<strong>Your Spotify library</strong>${s.done.pull ? "Pulled and on this machine." : "Not pulled yet."}
       <br><button class="btn small outline" id="use-demo">Try the demo library</button>`;
}

// ---------------------------------------------------------------- home

const STEP_INFO = {
  pull: { title: "Pull your library", spotify: "Liked Songs and the playlists you own. Read-only.", demo: "Generate a 3,000-song synthetic library with hidden genres." },
  tags: { title: "Fetch Last.fm tags", spotify: "Crowd tags for every song: the sorter's main signal. Resumable; roughly 10–30 min for 3,000 songs.", demo: "Included with the demo library." },
  sort: { title: "Sort", spotify: "Pick a model, score every unsorted liked song, and find new playlists.", demo: "Pick a model, score every unsorted liked song, and find new playlists." },
};

function greeting() {
  const h = new Date().getHours();
  return h < 5 ? "Good night" : h < 12 ? "Good morning" : h < 18 ? "Good afternoon" : "Good evening";
}

function stepState(name) {
  const s = App.session, j = s.jobs.steps[name];
  if (["running", "queued", "failed"].includes(j.status)) return j.status;
  if (j.status === "skipped" || s.done[name]) return "done";
  return "idle";
}

function renderHome() {
  const s = App.session;
  const who = s.mode === "demo" ? "" : `, ${esc(s.user?.name?.split(" ")[0] || "")}`;
  $("#home-hero").innerHTML = `<div class="eyebrow">${s.mode === "demo" ? "Demo library" : "Your Spotify"}</div>
    <h1>${greeting()}${who}</h1>
    <p class="lede">${App.ready
      ? "Your suggestions are ready. Sort the songs, name the new playlists, then see what your taste looks like."
      : "Three steps and your Liked Songs are ready to sort. Run them all, or one at a time."}</p>`;

  $("#steps").innerHTML = ["pull", "tags", "sort"].map((name, i) => {
    const st = stepState(name), j = s.jobs.steps[name], info = STEP_INFO[name];
    const label = { done: "Done", running: j.progress != null ? `Running · ${pct(j.progress)}` : "Running", queued: "Queued", failed: "Failed", idle: "Not run" }[st];
    let note = info[s.mode];
    if (name === "sort" && s.mode === "spotify" && !s.has_decision) note += " Uses tags + LSA until the Phase 1 experiments pick a featurizer.";
    const busy = st === "running" || st === "queued";
    const demoTags = s.mode === "demo" && name === "tags";
    return `<div class="step ${st}">
      <div class="step-top"><span class="step-num">${st === "done" ? "✓" : st === "failed" ? "!" : i + 1}</span>
        <div><h3>${info.title}</h3></div></div>
      <p>${esc(note)}</p>
      ${busy ? `<div class="bar ${j.progress == null ? "indeterminate" : ""}"><div style="width:${pct(j.progress || 0)}"></div></div>` : ""}
      ${j.lines.length && st !== "done" ? `<pre class="log">${esc(j.lines.slice(st === "failed" ? -8 : -3).join("\n"))}</pre>` : ""}
      <div class="row"><span class="status-pill ${st}">${label}</span>
        ${demoTags ? "" : busy ? (st === "running" ? `<button class="btn small outline" data-stop>Stop</button>` : "")
          : `<button class="btn small ${st === "done" ? "outline" : "primary"}" data-run="${name}">${st === "done" ? "Run again" : st === "failed" ? "Retry" : "Run"}</button>`}
      </div></div>`;
  }).join("");

  const allDone = ["pull", "tags", "sort"].every((n) => stepState(n) === "done");
  const run = $("#run-all");
  run.disabled = s.jobs.running;
  run.textContent = s.jobs.running ? "Running…" : allDone ? "Refresh everything" : "Run all steps";

  $("#shortcuts").innerHTML = App.ready && App.shortcuts ? `<div class="section-head"><h2>Jump back in</h2></div><div class="shelf">${App.shortcuts()}</div>` : "";
}

function stepsToRun() {
  const order = ["pull", "tags", "sort"];
  const first = order.findIndex((n) => stepState(n) !== "done");
  return first === -1 ? order : order.slice(first);
}

// ---------------------------------------------------------------- routing

function showView() {
  let name = location.hash.slice(1);
  if (!VIEWS.includes(name)) name = "home";
  const open = name === "home" || App.ready || (NEEDS_LIBRARY.includes(name) && App.session?.done.pull);
  if (!open) name = "home";
  App.view = name;
  $$(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.view === name));
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
  $("#scroll").scrollTop = 0;
  for (const fn of App.hooks.view) fn(name);
}
window.addEventListener("hashchange", showView);

// ---------------------------------------------------------------- events

document.addEventListener("click", guard(async (e) => {
  if (e.target.closest("#login-btn.disabled")) { e.preventDefault(); return; }
  if (e.target.closest("#demo-btn, #use-demo")) {
    await api("/api/mode", { mode: "demo" });
    history.replaceState(null, "", "/#home");
    App.ready = false;
    await refreshSession();
    return showView();
  }
  if (e.target.closest("#use-spotify")) {
    if (!App.session.configured) { await api("/api/mode", { mode: "spotify" }); App.ready = false; return refreshSession(); }
    await api("/api/mode", { mode: "spotify" });
    App.ready = false;
    const s = await refreshSession();
    if (!s.user) location.href = "/login";
    return showView();
  }
  if (e.target.closest("#logout")) {
    await api("/api/logout", {});
    App.ready = false;
    return refreshSession();
  }
  const runBtn = e.target.closest("[data-run]");
  if (runBtn) { await api("/api/run", { steps: [runBtn.dataset.run] }); return refreshSession(); }
  if (e.target.closest("[data-stop]")) { await api("/api/stop", {}); return refreshSession(); }
  if (e.target.closest("#run-all")) { await api("/api/run", { steps: stepsToRun() }); return refreshSession(); }
}));

guard(async () => {
  await refreshSession();
  showView();
})();

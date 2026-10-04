// Sorting views: Sort songs (review queue), New playlists, Your playlists, Apply.
// Shared helpers ($, esc, api, toast, guard, songs, pct) come from app.js.
"use strict";

const S = { meta: null, playlists: [], plById: {}, clusters: [], clById: {}, songs: [], decisions: {} };
const ui = { tier: "all", search: "", undecidedOnly: true, sel: null };

async function load() {
  const st = await api("/api/state");
  Object.assign(S, st);
  S.plById = Object.fromEntries(st.playlists.map((p) => [p.id, p]));
  S.clById = Object.fromEntries(st.clusters.map((c) => [c.id, c]));
  render();
}

// ---------------------------------------------------------------- helpers

function vibeText(v) {
  if (!v) return "";
  const parts = v.tags.length ? v.tags.slice(0, 3).map((t) => t.tag) : v.axes.map((a) => a.label);
  if (!parts.length && v.genre) parts.push(v.genre.split("---").pop());
  if (!parts.length && v.artists.length) parts.push(`like ${v.artists[0]}`);
  return parts.join(", ");
}

function vibeChips(v) {
  if (!v) return "";
  const chips = [
    ...v.tags.map((t) => `<span class="tag" title="${pct(t.share)} of songs, ${t.lift}x the library">${esc(t.tag)}</span>`),
    ...v.axes.map((a) => `<span class="tag axis">${esc(a.label)}</span>`),
  ];
  if (v.genre) chips.push(`<span class="tag axis">${esc(v.genre.split("---").pop())}</span>`);
  if (v.decade) chips.push(`<span class="tag">${esc(v.decade)}</span>`);
  return `<div class="tags">${chips.join("")}</div>`;
}

function decisionLabel(d) {
  if (!d) return "";
  if (d.action === "add") return "Add to " + d.playlist_ids.map((id) => S.plById[id]?.name || id).join(", ");
  if (d.action === "new") return "New playlist: " + (S.clById[d.cluster_id]?.name || d.cluster_id);
  if (d.action === "auto") return "Added to new playlist";
  return "Skipped";
}

function filtered() {
  const q = ui.search.trim().toLowerCase();
  return S.songs.filter((s) => {
    if (ui.tier !== "all" && s.tier !== ui.tier) return false;
    if (ui.undecidedOnly && S.decisions[s.track_id] && s.track_id !== ui.sel) return false;
    if (q && !(`${s.name} ${s.artists.join(" ")} ${s.album || ""}`.toLowerCase().includes(q))) return false;
    return true;
  });
}

// ---------------------------------------------------------------- render

function render() {
  const total = S.songs.length;
  const decided = S.songs.filter((s) => S.decisions[s.track_id]).length;
  $("#progress-bar").style.width = total ? `${(100 * decided) / total}%` : "0";
  $("#progress-text").textContent = `${decided.toLocaleString()} / ${total.toLocaleString()} reviewed`;
  $("#n-review").textContent = total - decided || "";
  $("#n-clusters").textContent = S.clusters.length || "";
  renderQueue();
  renderCard();
  if (App.view === "clusters") renderClusters();
  if (App.view === "playlists") renderPlaylists();
  if (App.view === "apply") guard(renderApply)();
  if (App.view === "home") renderHomeShelf();
  refreshPlanCount();
}

function renderQueue() {
  const list = filtered();
  if (!ui.sel || !list.some((s) => s.track_id === ui.sel)) ui.sel = list[0]?.track_id ?? null;
  const ol = $("#queue-list");
  if (!list.length) {
    ol.innerHTML = `<li class="queue-empty">${S.songs.length ? "Nothing left in this view. Nice." : "No unsorted liked songs."}</li>`;
  } else {
    // Render a window around the selection to keep long libraries snappy.
    const i = Math.max(0, list.findIndex((s) => s.track_id === ui.sel));
    const start = Math.max(0, i - 100), slice = list.slice(start, start + 300);
    ol.innerHTML = slice.map((s) => {
      const d = S.decisions[s.track_id];
      const badge = d ? `<span class="badge ${d.action === "skip" ? "skip" : "done"}">${d.action === "skip" ? "skip" : "done"}</span>`
                      : `<span class="badge ${s.tier}">${s.tier === "no_data" ? "no data" : s.tier}</span>`;
      return `<li data-id="${esc(s.track_id)}" class="${s.track_id === ui.sel ? "sel" : ""}">
        <span class="t">${esc(s.name)}</span><span class="s">${badge}</span>
        <span class="a">${esc(s.artists.join(", "))}</span></li>`;
    }).join("");
    $("li.sel", ol)?.scrollIntoView({ block: "nearest" });
  }
  const nConf = S.songs.filter((s) => s.tier === "confident" && s.suggestions.length && !S.decisions[s.track_id]).length;
  const bulk = $("#bulk");
  bulk.textContent = nConf ? `Accept all ${nConf} confident suggestions` : "No confident suggestions left";
  bulk.disabled = !nConf;
}

function renderCard() {
  const card = $("#card");
  const s = S.songs.find((x) => x.track_id === ui.sel);
  if (!s) {
    card.innerHTML = `<p class="muted">Pick a song on the left.</p>`;
    return;
  }
  const d = S.decisions[s.track_id];
  const chosen = new Set(d?.action === "add" ? d.playlist_ids : []);
  const player = s.preview_url
    ? `<audio controls preload="none" src="${esc(s.preview_url)}"></audio>`
    : S.meta.demo ? `<p class="muted">No audio in demo mode.</p>`
    : `<iframe loading="lazy" allow="encrypted-media" title="Spotify player" src="https://open.spotify.com/embed/track/${encodeURIComponent(s.track_id)}"></iframe>`;

  const tierHelp = {
    confident: "Held-out accuracy at this confidence was at or above the target.",
    suggested: "A reasonable guess; check it.",
    leftover: s.novel ? "Sounds unlike your playlists; probably a new playlist." : "Low confidence for every playlist.",
    no_data: "No tags or audio for this song, so no suggestion. Pick by hand.",
  }[s.tier];

  const sugg = s.suggestions.map((g, i) => {
    const p = S.plById[g.playlist_id];
    return `<button data-pl="${esc(g.playlist_id)}" class="${chosen.has(g.playlist_id) ? "chosen" : ""}">
      <kbd>${i + 1}</kbd><span class="name">${esc(p?.name || g.playlist_id)}</span>
      <span class="vibe">${esc(vibeText(p?.vibe))}</span>
      <span class="bar">${pct(g.p)}<span class="meter"><div style="width:${pct(g.p)}"></div></span></span></button>`;
  }).join("");

  const writable = S.playlists.filter((p) => p.writable);
  const clusterOpts = S.clusters.map((c) =>
    `<option value="${esc(c.id)}" ${c.id === (d?.cluster_id || s.cluster) ? "selected" : ""}>${esc(c.name)}</option>`).join("");

  card.innerHTML = `
    <span class="badge ${s.tier}" title="${esc(tierHelp)}">${s.tier === "no_data" ? "no data" : s.tier}</span>
    ${s.confidence != null ? `<span class="muted"> &nbsp;${pct(s.confidence)} sure</span>` : ""}
    <h1>${esc(s.name)}</h1>
    <p class="by">${esc(s.artists.join(", "))}${s.album ? ` · ${esc(s.album)}` : ""}${s.release_date ? ` · ${esc(String(s.release_date).slice(0, 4))}` : ""}</p>
    <div class="player">${player}</div>
    ${s.suggestions.length ? `<h3>Best fits</h3><div class="sugg">${sugg}</div>` : ""}
    <div class="actions">
      <div class="other">
        <input type="text" id="other-pl" list="pl-list" placeholder="Other playlist…" aria-label="Other playlist">
        <datalist id="pl-list">${writable.map((p) => `<option value="${esc(p.name)}">`).join("")}</datalist>
        <button class="btn" id="other-add">Add</button>
      </div>
    </div>
    <div class="actions">
      ${S.clusters.length ? `<select id="cluster-pick" class="btn" aria-label="New playlist">${clusterOpts}</select>
        <button class="btn" id="to-new"><kbd>N</kbd> Send to new playlist</button>` : ""}
      <button class="btn" id="skip"><kbd>S</kbd> Skip</button>
    </div>
    ${d ? `<div class="status"><span class="badge done">decided</span> ${esc(decisionLabel(d))}
      ${d.applied_at ? `<span class="muted">· applied ${esc(d.applied_at)}</span>` : `<button class="btn small" id="undo"><kbd>Z</kbd> Undo</button>`}</div>` : ""}
    <p class="muted" style="margin-top:16px;font-size:13px">${esc(tierHelp)}</p>`;
}

function renderClusters() {
  const grid = $("#cluster-grid");
  if (!S.clusters.length) {
    grid.innerHTML = `<p class="muted">No clusters: every unsorted song fit an existing playlist well enough, or there were too few leftovers.</p>`;
    return;
  }
  grid.innerHTML = S.clusters.map((c) => {
    const members = c.members.map((t) => S.songs.find((s) => s.track_id === t)).filter(Boolean);
    return `<div class="tile ${c.approved ? "on" : ""}" data-cid="${esc(c.id)}">
      <label class="toggle"><input type="checkbox" class="approve" ${c.approved ? "checked" : ""}>
        ${c.playlist_id ? "Created on Spotify" : "Create this playlist"}</label>
      <input type="text" class="cname" value="${esc(c.name)}" aria-label="Playlist name" ${c.playlist_id ? "disabled" : ""}>
      <div class="meta">${songs(members.length)}${c.vibe.artists.length ? ` · ${esc(c.vibe.artists.join(", "))}` : ""}</div>
      ${vibeChips(c.vibe)}
      <ul class="songs">${members.map((s) => `<li><span>${esc(s.name)} · <span class="muted">${esc(s.artists.join(", "))}</span></span>
        <button class="btn link small drop" data-id="${esc(s.track_id)}" title="Leave this song out">remove</button></li>`).join("")}</ul>
    </div>`;
  }).join("");
}

function renderPlaylists() {
  const pending = {};
  for (const d of Object.values(S.decisions)) {
    if (d.action === "add" && !d.applied_at) for (const id of d.playlist_ids) pending[id] = (pending[id] || 0) + 1;
  }
  $("#playlist-grid").innerHTML = [...S.playlists].sort((a, b) => b.size - a.size).map((p) => `
    <div class="tile">
      <h4>${esc(p.name)}</h4>
      <div class="meta">${songs(p.size)}${pending[p.id] ? ` · <strong>+${pending[p.id]} pending</strong>` : ""}
        ${!p.scored ? " · too small to learn from (under 10 songs)" : ""}${!p.writable ? " · read-only (not yours)" : ""}</div>
      ${vibeChips(p.vibe)}
      ${p.vibe.artists.length ? `<div class="meta">Top artists: ${esc(p.vibe.artists.join(", "))}</div>` : ""}
    </div>`).join("");
}

async function refreshPlanCount() {
  try {
    const { plan } = await api("/api/plan");
    $("#n-plan").textContent = plan.n_songs || "";
  } catch { /* shown on the apply tab */ }
}

async function renderApply() {
  const { plan, demo, applied } = await api("/api/plan");
  const rows = [
    ...plan.create.map((c) => `<li><span>${c.playlist_id ? "Add to new playlist" : "Create playlist"} <strong>${esc(c.name)}</strong></span><span>${songs(c.track_ids.length)}</span></li>`),
    ...plan.add.map((a) => `<li><span>Add to <strong>${esc(a.name)}</strong></span><span>${songs(a.track_ids.length)}</span></li>`),
  ];
  $("#plan").innerHTML = rows.length ? `<ul class="plan-list">${rows.join("")}</ul>`
    : `<p class="muted">Nothing yet. Review some songs or switch on a new playlist.</p>`;
  const btn = $("#apply-btn");
  btn.disabled = demo || !plan.n_songs;
  btn.textContent = demo ? "Apply to Spotify (off in demo mode)" : `Apply ${plan.n_songs} changes to Spotify`;
  $("#model-info").innerHTML = S.meta.spaces.map((sp) =>
    `<p>Space <code>${esc(sp.run_id)}</code> · model <code>${esc(sp.model)}</code> · held-out top-1 ${sp.top1.toFixed(1)}%, top-3 ${sp.top3.toFixed(1)}%
     · confident tier from ${sp.calibration.confident_threshold == null ? "n/a" : pct(sp.calibration.confident_threshold)}
     at ${pct(sp.calibration.target_precision)} precision</p>`).join("") +
    `<p>Suggestions generated ${esc(S.meta.generated_at)} from library pulled ${esc(S.meta.pulled_at)}.</p>`;
  $("#applied").innerHTML = applied.length ? applied.map((n) =>
    `<li><code>${esc(n)}</code> <button class="btn small undo-apply" data-name="${esc(n)}">Undo</button></li>`).join("")
    : `<li class="muted">None yet.</li>`;
}

// ---------------------------------------------------------------- actions

function advance() {
  const list = filtered();
  const i = list.findIndex((s) => s.track_id === ui.sel);
  const next = list.slice(i + 1).find((s) => !S.decisions[s.track_id]) || list.find((s) => !S.decisions[s.track_id] && s.track_id !== ui.sel);
  if (next) ui.sel = next.track_id;
}

async function decide(body) {
  const { decision } = await api("/api/decide", body);
  S.decisions[body.track_id] = decision;
  advance();
  render();
}

const addTo = guard((pid) => decide({ track_id: ui.sel, action: "add", playlist_ids: [pid] }));
const skip = guard(() => decide({ track_id: ui.sel, action: "skip" }));
const toNew = guard(() => {
  const cid = $("#cluster-pick")?.value;
  if (!cid) throw new Error("No new-playlist clusters to send it to.");
  return decide({ track_id: ui.sel, action: "new", cluster_id: cid });
});
const undo = guard(async () => {
  if (!S.decisions[ui.sel] || S.decisions[ui.sel].applied_at) return;
  await api("/api/undo", { track_id: ui.sel });
  delete S.decisions[ui.sel];
  render();
});

function move(delta) {
  const list = filtered();
  const i = list.findIndex((s) => s.track_id === ui.sel);
  const j = Math.min(list.length - 1, Math.max(0, i + delta));
  if (list[j]) { ui.sel = list[j].track_id; renderQueue(); renderCard(); }
}

// ---------------------------------------------------------------- events

document.addEventListener("click", guard(async (e) => {
  const li = e.target.closest("#queue-list li[data-id]");
  if (li) { ui.sel = li.dataset.id; renderQueue(); renderCard(); return; }
  const chip = e.target.closest("#tier-filter .chip");
  if (chip) {
    ui.tier = chip.dataset.tier;
    document.querySelectorAll("#tier-filter .chip").forEach((c) => c.classList.toggle("active", c === chip));
    renderQueue(); renderCard();
    return;
  }
  const sug = e.target.closest(".sugg button");
  if (sug) return addTo(sug.dataset.pl);
  if (e.target.closest("#skip")) return skip();
  if (e.target.closest("#to-new")) return toNew();
  if (e.target.closest("#undo")) return undo();
  if (e.target.closest("#other-add")) return addOther();
  if (e.target.closest("#bulk")) {
    const { accepted } = await api("/api/bulk_accept", { tier: "confident" });
    toast(`Accepted ${accepted} confident suggestions`);
    return load();
  }
  const drop = e.target.closest(".drop");
  if (drop) {
    await api("/api/decide", { track_id: drop.dataset.id, action: "skip" });
    return load();
  }
  if (e.target.closest("#apply-btn")) {
    const { plan } = await api("/api/plan");
    if (!confirm(`Apply ${plan.n_songs} changes to your Spotify library?\n\nThis adds songs and creates private playlists. You can undo it afterwards.`)) return;
    e.target.disabled = true;
    e.target.textContent = "Applying… (a Spotify login window may open)";
    const r = await api("/api/apply", {});
    toast(r.log.errors.length ? `Applied with ${r.log.errors.length} errors: ${r.log.errors[0].error}` : "Applied to Spotify", r.log.errors.length > 0);
    return load();
  }
  const ua = e.target.closest(".undo-apply");
  if (ua) {
    if (!confirm(`Undo ${ua.dataset.name}? This removes the songs it added and unfollows the playlists it created.`)) return;
    const { result } = await api("/api/undo_apply", { changelog: ua.dataset.name });
    toast(`Removed ${result.removed} songs, unfollowed ${result.unfollowed} playlists` + (result.errors.length ? ` (${result.errors.length} errors)` : ""), result.errors.length > 0);
    return load();
  }
}));

document.addEventListener("change", guard(async (e) => {
  const tile = e.target.closest(".tile[data-cid]");
  if (tile && e.target.matches(".approve")) {
    await api("/api/cluster", { cluster_id: tile.dataset.cid, approved: e.target.checked });
    return load();
  }
  if (tile && e.target.matches(".cname")) {
    await api("/api/cluster", { cluster_id: tile.dataset.cid, name: e.target.value });
    toast("Renamed");
    return load();
  }
  if (e.target.id === "undecided-only") { ui.undecidedOnly = e.target.checked; renderQueue(); renderCard(); }
}));

$("#search").addEventListener("input", (e) => { ui.search = e.target.value; renderQueue(); renderCard(); });

const addOther = guard(() => {
  const name = $("#other-pl").value.trim().toLowerCase();
  const p = S.playlists.find((x) => x.writable && x.name.toLowerCase() === name);
  if (!p) throw new Error("Pick one of your playlists from the list.");
  return addTo(p.id);
});

document.addEventListener("keydown", (e) => {
  if (e.target.matches("input, select, textarea")) {
    if (e.key === "Enter" && e.target.id === "other-pl") addOther();
    if (e.key === "Escape") e.target.blur();
    return;
  }
  if (App.view !== "review" || e.metaKey || e.ctrlKey || e.altKey) return;
  const s = S.songs.find((x) => x.track_id === ui.sel);
  const k = e.key.toLowerCase();
  if (["1", "2", "3"].includes(k) && s?.suggestions[+k - 1]) addTo(s.suggestions[+k - 1].playlist_id);
  else if (k === "s" && s) skip();
  else if (k === "n" && s) toNew();
  else if (k === "z") undo();
  else if (k === "j" || e.key === "ArrowDown") move(1);
  else if (k === "k" || e.key === "ArrowUp") move(-1);
  else if (k === "p") { const a = $("#card audio"); if (a) a.paused ? a.play() : a.pause(); }
  else if (k === "/") $("#search").focus();
  else return;
  e.preventDefault();
});

// ---------------------------------------------------------------- hooks into the shell

function renderHomeShelf() {
  if (App.session) renderHome();          // app.js: redraws the steps and the "Jump back in" shelf
}

App.shortcuts = () => {
  if (!S.meta) return "";
  const left = S.songs.filter((s) => !S.decisions[s.track_id]).length;
  const on = S.clusters.filter((c) => c.approved).length;
  const tile = (href, art, bg, title, sub) =>
    `<a class="tile-link" href="#${href}"><div class="art" style="background:${bg}">${art}</div><h4>${esc(title)}</h4><p>${esc(sub)}</p></a>`;
  return [
    tile("review", left.toLocaleString(), "linear-gradient(135deg,#1ed760,#0b5e2a)", "Sort songs", `${songs(left)} left to review`),
    tile("clusters", "✦", "linear-gradient(135deg,#9085e9,#3b2f8f)", "New playlists", `${S.clusters.length} found · ${on} switched on`),
    tile("gallery", "◐", "linear-gradient(135deg,#d95926,#9085e9)", "Taste gallery", "Your library, visualized"),
    tile("playlists", S.playlists.length, "linear-gradient(135deg,#3987e5,#123d73)", "Your playlists", "What each one sounds like"),
    tile("apply", "→", "linear-gradient(135deg,#c98500,#6b3d00)", "Apply", "Send approved changes to Spotify"),
  ].join("");
};

App.hooks.ready.push(load);
App.hooks.view.push((name) => {
  if (!S.meta) return;
  if (name === "review") { renderQueue(); renderCard(); }
  else render();
});

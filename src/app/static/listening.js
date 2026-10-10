// Listening: your streaming history over time, genre mix, listening calendar, taste radars and the
// artist web, drawn with d3 from /api/listening/*. Drawing helpers come from gallery.js (App.viz).
// Colors: genres take the categorical slots in their overall order, so a genre keeps its color in every
// chart and time range; picked lines keep the slot they were given; two webs are green and violet.
"use strict";

(() => {
  const V = () => App.viz;
  // Dark-mode categorical slots, validated against the #181818 card surface.
  const PAL = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
  const OTHER = "#5c5c5c";
  const ALL = "__all__";
  const DIMS = ["artist", "album", "track", "genre", "playlist"];
  const DIM = { artist: ["Artist", "Artists"], album: ["Album", "Albums"], track: ["Song", "Songs"], genre: ["Genre", "Genres"], playlist: ["Playlist", "Playlists"] };
  const TZ = (() => { try { return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC"; } catch { return "UTC"; } })();
  const DAY = 864e5;

  const L = {
    meta: null, tables: new Set(), data: {}, tok: {}, importOpen: false,
    line: { dim: "artist", bucket: "month", items: null, slots: {} },
    pie: { preset: "12m", start: null, end: null },
    cal: { year: null, dim: "", item: "" },
    radar: { a: null, b: null },
    chord: { genre: null },
  };

  // ---------------------------------------------------------------- small helpers

  async function get(path, params = {}) {
    const q = new URLSearchParams({ tz: TZ });
    for (const [k, v] of Object.entries(params)) {
      if (Array.isArray(v)) v.forEach((x) => q.append(k, x));
      else if (v != null && v !== "") q.set(k, v);
    }
    return api(`/api/listening/${path}?${q}`);
  }
  const hrs = (h) => (h < 1 ? `${Math.round(h * 60)} min` : `${h < 10 ? h.toFixed(1) : V().fmt(Math.round(h))} h`);
  const parseDay = (s) => { const [y, m, d] = s.split("-").map(Number); return new Date(y, m - 1, d); };
  const iso = (dt) => `${dt.getFullYear()}-${String(dt.getMonth() + 1).padStart(2, "0")}-${String(dt.getDate()).padStart(2, "0")}`;
  const addDays = (s, n) => iso(new Date(parseDay(s).getTime() + n * DAY));
  const niceDate = (s, opts = { month: "short", day: "numeric", year: "numeric" }) => parseDay(s).toLocaleDateString(undefined, opts);
  const label = (k) => (k === ALL ? "All listening" : k);

  function genreColor(g) {
    const i = L.meta?.genre_rank.indexOf(g) ?? -1;
    return i >= 0 && i < PAL.length ? PAL[i] : OTHER;
  }
  // For a short list of genres (a pie): overall-top genres keep their slot, the others take free slots.
  function genreColors(list) {
    const used = new Set(list.map(genreColor).filter((c) => c !== OTHER));
    const free = PAL.filter((c) => !used.has(c));
    return Object.fromEntries(list.map((g) => [g, genreColor(g) !== OTHER ? genreColor(g) : free.shift() || OTHER]));
  }

  function chips(name, options, current) {
    return `<div class="chips" role="group" aria-label="${esc(name)}">${options.map(([v, t]) =>
      `<button class="chip ${String(v) === String(current) ? "active" : ""}" data-${name}="${esc(v)}">${esc(t)}</button>`).join("")}</div>`;
  }
  function on(el, sel, fn, ev = "click") {
    el.addEventListener(ev, (e) => { const t = e.target.closest(sel); if (t && el.contains(t)) fn(t, e); });
  }
  function legend(el, items) {
    const lg = document.createElement("div");
    lg.className = "legend";
    lg.innerHTML = items.map(([c, t]) => `<span><span class="sw" style="background:${c}"></span>${esc(t)}</span>`).join("");
    el.append(lg);
  }
  function axisStyle(g) { g.select(".domain").remove(); g.selectAll("line").attr("stroke", V().C.line); }

  // Typeahead over a dimension's items (by listening hours), into a <datalist>.
  function itemSearch(input, dim, onPick) {
    const dl = document.createElement("datalist");
    dl.id = `dl-${Math.random().toString(36).slice(2)}`;
    input.setAttribute("list", dl.id);
    input.after(dl);
    const fill = (items) => { dl.innerHTML = items.map(([k, h]) => `<option value="${esc(k)}">${esc(hrs(h))}</option>`).join(""); };
    fill((L.meta.items[dim] || []).slice(0, 50));
    let t, last = [];
    input.addEventListener("input", () => {
      clearTimeout(t);
      const exact = last.find(([k]) => k === input.value) || (L.meta.items[dim] || []).find(([k]) => k === input.value);
      if (exact) return onPick(exact[0]);
      t = setTimeout(guard(async () => { last = (await get("search", { dim, q: input.value })).items; fill(last); }), 180);
    });
    input.addEventListener("keydown", (e) => {
      if (e.key !== "Enter") return;
      const hit = [...dl.options].find((o) => o.value.toLowerCase() === input.value.toLowerCase()) || dl.options[0];
      if (hit) onPick(hit.value);
    });
  }

  // ---------------------------------------------------------------- 1. hours over time

  function periodLabel(s, bucket) {
    if (bucket === "year") return s.slice(0, 4);
    if (bucket === "month") return niceDate(s, { month: "short", year: "numeric" });
    return `Week of ${niceDate(s)}`;
  }

  const timeline = {
    id: "timeline", wide: true, history: true, title: "Hours over time",
    blurb: "How long you spent listening, per week, month or year. Add artists, albums, songs, genres or playlists to compare them.",
    controls(el) {
      const s = L.line;
      const picked = s.items || [];
      el.innerHTML = `${chips("dim", DIMS.map((d) => [d, DIM[d][1]]), s.dim)}
        ${chips("bucket", [["week", "Weekly"], ["month", "Monthly"], ["year", "Yearly"]], s.bucket)}
        <div class="adder"><input type="search" placeholder="Add ${DIM[s.dim][0].toLowerCase()}…" aria-label="Add ${DIM[s.dim][0].toLowerCase()}" ${picked.length >= 6 ? "disabled" : ""}></div>
        <div class="chips picked">${picked.map((k) => `<button class="chip removable" data-rm="${esc(k)}" title="Remove">
          <span class="sw" style="background:${lineColor(k)}"></span>${esc(V().trunc(label(k), 32))}<span class="x" aria-hidden="true">×</span></button>`).join("")}
          ${picked.includes(ALL) ? "" : `<button class="chip ghost" data-add-all>+ All listening</button>`}</div>`;
      on(el, "[data-dim]", (b) => { if (s.dim !== b.dataset.dim) { s.dim = b.dataset.dim; s.items = null; s.slots = {}; refresh(timeline); } });
      on(el, "[data-bucket]", (b) => { s.bucket = b.dataset.bucket; refresh(timeline); });
      on(el, "[data-rm]", (b) => { s.items = picked.filter((k) => k !== b.dataset.rm); delete s.slots[b.dataset.rm]; refresh(timeline); });
      on(el, "[data-add-all]", () => addLine(ALL));
      itemSearch($("input", el), s.dim, addLine);
    },
    load: () => get("series", { dim: L.line.dim, bucket: L.line.bucket, item: L.line.items || [] }),
    loaded(d) {
      if (L.line.items === null) { L.line.items = d.series.map((x) => x.key); d.series.forEach((x) => lineColor(x.key)); return true; }
    },
    render(el, d, w) {
      const { C } = V();
      const ser = d.series.filter((x) => L.line.items.includes(x.key));
      if (!ser.length) return V().empty(el, `Add ${DIM[d.dim][0].toLowerCase()}s to plot.`);
      const dates = d.periods.map(parseDay);
      const direct = ser.length <= 4 && w > 560;
      const h = 300, m = { t: 14, r: direct ? 150 : 16, b: 28, l: 46 };
      const x = d3.scaleTime(d3.extent(dates), [m.l, w - m.r]);
      const y = d3.scaleLinear([0, d3.max(ser, (s) => d3.max(s.hours)) || 1], [h - m.b, m.t]).nice();
      const svg = d3.select(el).append("svg").attr("width", w).attr("height", h)
        .attr("role", "img").attr("aria-label", "Line chart of listening hours over time; use the table view for the numbers.");
      svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`)
        .call(d3.axisLeft(y).ticks(5).tickFormat((v) => `${v} h`).tickSize(-(w - m.l - m.r))).call(axisStyle);
      svg.append("g").attr("class", "axis").attr("transform", `translate(0,${h - m.b})`)
        .call(d3.axisBottom(x).ticks(Math.max(2, Math.floor((w - m.l - m.r) / 90))).tickSizeOuter(0));
      const line = d3.line((v, i) => x(dates[i]), (v) => y(v)).curve(d3.curveMonotoneX);
      for (const s of ser) {
        svg.append("path").datum(s.hours).attr("fill", "none").attr("stroke", lineColor(s.key)).attr("stroke-width", 2)
          .attr("stroke-linejoin", "round").attr("stroke-linecap", "round").attr("d", line);
      }
      if (direct) {
        // End labels, nudged apart so they never overlap.
        const ends = ser.map((s) => ({ s, y: y(s.hours.at(-1)) })).sort((a, b) => a.y - b.y);
        for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 15);
        for (const e of ends) {
          svg.append("circle").attr("cx", x(dates.at(-1)) + 8).attr("cy", e.y).attr("r", 4).attr("fill", lineColor(e.s.key));
          svg.append("text").attr("x", x(dates.at(-1)) + 16).attr("y", e.y + 4).attr("fill", C.muted).attr("font-size", 12)
            .text(V().trunc(label(e.s.key), 20));
        }
      }
      // Crosshair + one tooltip for every line at the hovered period.
      const cross = svg.append("line").attr("y1", m.t).attr("y2", h - m.b).attr("stroke", C.muted).attr("stroke-dasharray", "3 3").attr("visibility", "hidden");
      const dots = svg.append("g");
      svg.on("mousemove", (e) => {
        const [mx] = d3.pointer(e);
        if (mx < m.l || mx > w - m.r) { cross.attr("visibility", "hidden"); dots.selectAll("*").remove(); return V().untip(); }
        const i = d3.minIndex(dates, (dt) => Math.abs(x(dt) - mx));
        cross.attr("x1", x(dates[i])).attr("x2", x(dates[i])).attr("visibility", "visible");
        dots.selectAll("circle").data(ser).join("circle").attr("cx", x(dates[i])).attr("cy", (s) => y(s.hours[i])).attr("r", 4)
          .attr("fill", (s) => lineColor(s.key)).attr("stroke", C.surface).attr("stroke-width", 2);
        const rows = [...ser].sort((a, b) => b.hours[i] - a.hours[i]).map((s) =>
          `<div class="s"><span class="sw" style="background:${lineColor(s.key)}"></span>${esc(V().trunc(label(s.key), 30))}: <strong>${hrs(s.hours[i])}</strong></div>`).join("");
        V().tip(e, `<div class="t">${esc(periodLabel(d.periods[i], d.bucket))}</div>${rows}`);
      }).on("mouseleave", () => { cross.attr("visibility", "hidden"); dots.selectAll("*").remove(); V().untip(); });
      if (ser.length >= 2) legend(el, ser.map((s) => [lineColor(s.key), label(s.key)]));
      const top = ser[0], peak = d3.maxIndex(top.hours);
      V().note(el, `<strong>${esc(label(top.key))}</strong> peaked in ${esc(periodLabel(d.periods[peak], d.bucket).replace("Week of", "the week of"))}
        at ${hrs(top.hours[peak])}, ${hrs(d3.sum(top.hours))} in all.${d.dim === "playlist" ? " A song in two playlists counts toward both." : ""}`);
    },
    table: (d) => ({ cols: ["Period", ...d.series.map((s) => label(s.key))], rows: d.periods.map((p, i) => [periodLabel(p, d.bucket), ...d.series.map((s) => s.hours[i].toFixed(2))]) }),
  };

  function lineColor(k) {
    if (k === ALL) return "#e8e8e8";
    const s = L.line.slots;
    if (s[k] == null) {
      const taken = new Set(Object.values(s));
      s[k] = PAL.findIndex((_, i) => !taken.has(i));
    }
    return PAL[s[k]] || OTHER;
  }
  function addLine(k) {
    const s = L.line;
    s.items = s.items || [];
    if (!s.items.includes(k) && s.items.length < 6) { s.items.push(k); lineColor(k); }
    refresh(timeline);
  }

  // ---------------------------------------------------------------- 2. genre mix

  function presetRange(p) {
    const last = L.meta.last, first = L.meta.first, y = +last.slice(0, 4);
    const r = {
      "30d": [addDays(last, -29), last], "6m": [addDays(last, -182), last], "12m": [addDays(last, -364), last],
      ytd: [`${y}-01-01`, last], all: [first, last],
    }[p];
    if (r) return r;
    if (/^\d{4}$/.test(p)) return [`${p}-01-01`, `${p}-12-31`];
    return [first, last];
  }
  function rangeControls(el, name, st, onChange, compact = false) {
    const presets = [["30d", "30 days"], ["6m", "6 months"], ["12m", "12 months"], ...L.meta.years.slice(-4).map((y) => [String(y), String(y)]), ["all", "All time"]];
    const pick = compact
      ? `<select aria-label="Period">${presets.map(([v, t]) => `<option value="${v}" ${v === st.preset ? "selected" : ""}>${t}</option>`).join("")}
          ${st.preset ? "" : `<option selected disabled>Custom</option>`}</select>`
      : chips(name, presets, st.preset);
    el.insertAdjacentHTML("beforeend", `${pick}
      <div class="dates"><input type="date" aria-label="From" value="${st.start}" min="${L.meta.first}" max="${L.meta.last}">
        <span class="muted">to</span><input type="date" aria-label="To" value="${st.end}" min="${L.meta.first}" max="${L.meta.last}"></div>`);
    const choose = (p) => { st.preset = p; [st.start, st.end] = presetRange(p); onChange(); };
    on(el, `[data-${name}]`, (b) => choose(b.dataset[name]));
    if (compact) $("select", el).addEventListener("change", (e) => choose(e.target.value));
    const [a, b] = $$(".dates input", el).slice(-2);
    const set = () => { if (a.value && b.value && a.value <= b.value) { st.preset = null; st.start = a.value; st.end = b.value; onChange(); } };
    a.addEventListener("change", set); b.addEventListener("change", set);
  }

  const genreMix = {
    id: "genres", history: true, title: "Genre mix",
    blurb: "Your most-played genres in a time frame, by hours. The range is relative to the end of your history.",
    defaults() { if (!L.pie.start) [L.pie.start, L.pie.end] = presetRange(L.pie.preset); },
    controls(el) { rangeControls(el, "pie", L.pie, () => refresh(genreMix)); },
    load: () => get("share", { start: L.pie.start, end: L.pie.end }),
    render(el, d, w) {
      const { C } = V();
      const tagged = d.total - d.untagged;
      if (!d.genres.length) return V().empty(el, d.total ? "None of these plays have a genre yet. Run the Tags step on Home." : "No listening in this range.");
      const top = d.genres.slice(0, 8);
      const rest = d.genres.slice(8);
      const slices = top.map(([g, h, a]) => ({ g, h, a }));
      if (rest.length) slices.push({ g: "Other", h: d3.sum(rest, (r) => r[1]), a: rest.slice(0, 4).map((r) => r[0]), other: true });
      const col = genreColors(top.map((t) => t[0]));
      const color = (s) => (s.other ? OTHER : col[s.g]);
      const wrap = document.createElement("div");
      wrap.className = "pie-wrap";
      el.append(wrap);
      const size = Math.min(300, w);
      const R = size / 2 - 4;
      const svg = d3.select(wrap).append("svg").attr("width", size).attr("height", size)
        .attr("role", "img").attr("aria-label", "Donut chart of listening hours by genre; the legend and table list the values.");
      const g = svg.append("g").attr("transform", `translate(${size / 2},${size / 2})`);
      const arcs = d3.pie().value((s) => s.h).sort(null)(slices);
      const arc = d3.arc().innerRadius(R * 0.6).outerRadius(R).cornerRadius(4);
      const big = d3.arc().innerRadius(R * 0.6).outerRadius(R + 4).cornerRadius(4);
      g.selectAll("path").data(arcs).join("path").attr("d", arc).attr("fill", (a) => color(a.data))
        .attr("stroke", C.surface).attr("stroke-width", 2)
        .on("mousemove", function (e, a) {
          d3.select(this).attr("d", big);
          V().tip(e, `<div class="t">${esc(a.data.g)}</div><div class="s">${hrs(a.data.h)} · ${pct(a.data.h / tagged)}</div>
            ${a.data.a.length ? `<div class="s">${a.data.other ? "Includes" : "Mostly"} ${esc(a.data.a.join(", "))}${a.data.other && rest.length > 4 ? "…" : ""}</div>` : ""}`);
        })
        .on("mouseleave", function () { d3.select(this).attr("d", arc); V().untip(); });
      g.append("text").attr("text-anchor", "middle").attr("y", -2).attr("fill", C.ink).attr("font-size", 24).attr("font-weight", 800).text(hrs(tagged));
      g.append("text").attr("text-anchor", "middle").attr("y", 18).attr("fill", C.muted).attr("font-size", 12).text(`${niceDate(d.start, { month: "short", year: "numeric" })} – ${niceDate(d.end, { month: "short", year: "numeric" })}`);
      const list = document.createElement("ol");
      list.className = "pie-legend";
      list.innerHTML = slices.map((s) => `<li><span class="sw" style="background:${color(s)}"></span><span class="n">${esc(s.g)}</span><span class="v">${pct(s.h / tagged)}</span></li>`).join("");
      wrap.append(list);
      V().note(el, `<strong>${esc(top[0][0])}</strong> was ${pct(top[0][1] / tagged)} of your listening here${top[1] ? `, then ${esc(top[1][0])} (${pct(top[1][1] / tagged)})` : ""}.
        ${rest.length ? `The other ${rest.length} genres share the remaining ${pct(slices.at(-1).h / tagged)}.` : ""}
        ${d.untagged / d.total > 0.05 ? `${pct(d.untagged / d.total)} of these hours have no genre yet; running the Tags step on Home fills them in.` : ""}`);
    },
    table: (d) => {
      const tagged = d.total - d.untagged;
      return { cols: ["Genre", "Hours", "Share"], rows: d.genres.map(([g, h]) => [g, h.toFixed(1), pct(h / tagged)]) };
    },
  };

  // ---------------------------------------------------------------- 3. listening calendar

  const calendar = {
    id: "calendar", wide: true, history: true, title: "Listening calendar",
    blurb: "Every day of the year, shaded by how long you listened. Filter to one artist, album, song, genre or playlist.",
    defaults() { if (!L.meta.years.includes(L.cal.year)) L.cal.year = L.meta.years.at(-1); },
    controls(el) {
      const s = L.cal;
      el.innerHTML = `${chips("year", L.meta.years.map((y) => [y, y]), s.year)}
        <select aria-label="Filter by">${[["", "All listening"], ...DIMS.map((d) => [d, DIM[d][0]])].map(([v, t]) => `<option value="${v}" ${s.dim === v ? "selected" : ""}>${t}</option>`).join("")}</select>
        ${s.dim ? `<div class="adder"><input type="search" placeholder="Pick ${DIM[s.dim][0].toLowerCase()}…" aria-label="Pick ${DIM[s.dim][0].toLowerCase()}" value="${esc(s.item)}"></div>` : ""}`;
      on(el, "[data-year]", (b) => { s.year = +b.dataset.year; refresh(calendar); });
      $("select", el).addEventListener("change", (e) => { s.dim = e.target.value; s.item = s.dim ? (L.meta.items[s.dim][0] || [""])[0] : ""; refresh(calendar); });
      if (s.dim) itemSearch($("input", el), s.dim, (k) => { s.item = k; refresh(calendar); });
    },
    load: () => get("calendar", { year: L.cal.year, dim: L.cal.dim, item: L.cal.item }),
    render(el, d, w) {
      const { C, seq } = V();
      const byDay = new Map(d.days.map((r) => [r[0], r]));
      const jan1 = new Date(d.year, 0, 1);
      const start = new Date(jan1.getTime() - jan1.getDay() * DAY);     // the Sunday on or before Jan 1
      const nDays = Math.round((new Date(d.year + 1, 0, 1) - jan1) / DAY);
      const weekOf = (dt) => Math.floor(Math.round((dt - start) / DAY) / 7);      // round: DST days are 23/25 h
      const left = 30, top = 20;
      const cell = Math.max(8, Math.min(18, Math.floor((w - left - 4) / 54)));
      const weeks = Math.ceil((jan1.getDay() + nDays) / 7);
      const vals = d.days.map((r) => r[1]).filter((v) => v > 0).sort(d3.ascending);
      const cuts = [0.25, 0.5, 0.75].map((q) => d3.quantile(vals, q) || 0);
      const level = (v) => (v <= 0 ? 0 : 1 + cuts.filter((c) => v > c).length);       // 0 = none, 1..4
      const fill = (lv) => (lv ? seq(lv / 4) : C.empty);
      const svg = d3.select(el).append("svg").attr("width", left + weeks * cell + 4).attr("height", top + 7 * cell + 4)
        .attr("role", "img").attr("aria-label", `Calendar heatmap of listening in ${d.year}; use the table view for the numbers.`);
      const cells = [];
      for (let i = 0; i < nDays; i++) {
        const dt = new Date(d.year, 0, 1 + i);
        const wk = weekOf(dt);
        const r = byDay.get(iso(dt));
        cells.push({ dt, wk, dow: dt.getDay(), h: r ? r[1] : 0, top: r ? r[2] : null });
      }
      svg.append("g").selectAll("rect").data(cells).join("rect")
        .attr("x", (c) => left + c.wk * cell).attr("y", (c) => top + c.dow * cell)
        .attr("width", cell - 2).attr("height", cell - 2).attr("rx", 2)
        .attr("fill", (c) => fill(level(c.h)))
        .on("mousemove", (e, c) => V().tip(e, `<div class="t">${c.dt.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric", year: "numeric" })}</div>
          <div class="s">${c.h ? hrs(c.h) : "No listening"}</div>${c.top ? `<div class="s">Top ${d.top_kind === "track" ? "song" : "artist"}: ${esc(c.top)}</div>` : ""}`))
        .on("mouseleave", V().untip);
      for (let mo = 0; mo < 12; mo++) {
        const first = new Date(d.year, mo, 1);
        const wk = weekOf(first) + (first.getDay() ? 1 : 0);                   // first full week, like GitHub
        svg.append("text").attr("x", left + wk * cell).attr("y", top - 6).attr("fill", C.muted).attr("font-size", 11)
          .text(first.toLocaleDateString(undefined, { month: "short" }));
      }
      [[1, "Mon"], [3, "Wed"], [5, "Fri"]].forEach(([r, t]) => svg.append("text").attr("x", 0).attr("y", top + r * cell + cell - 4)
        .attr("fill", C.muted).attr("font-size", Math.min(11, cell)).text(t));
      const lg = document.createElement("div");
      lg.className = "legend cal-legend";
      lg.innerHTML = `<span>Less</span><span class="cells">${[0, 1, 2, 3, 4].map((lv) => `<span class="cell" style="background:${fill(lv)}"></span>`).join("")}</span><span>More</span>
        <span class="muted">${cuts[2] ? `Darkest: over ${hrs(cuts[2])} a day` : ""}</span>`;
      el.append(lg);
      if (!vals.length) return V().note(el, `No listening${d.item ? ` to ${esc(d.item)}` : ""} in ${d.year}.`);
      const best = cells.reduce((a, b) => (b.h > a.h ? b : a));
      let streak = 0, run = 0;
      for (const c of cells) { run = c.h > 0 ? run + 1 : 0; streak = Math.max(streak, run); }
      V().note(el, `${hrs(d.total)} on <strong>${vals.length}</strong> days${d.item ? ` of <strong>${esc(d.item)}</strong>` : ""} in ${d.year}.
        Biggest day: <strong>${best.dt.toLocaleDateString(undefined, { month: "long", day: "numeric" })}</strong> (${hrs(best.h)}). Longest streak: ${streak} day${streak === 1 ? "" : "s"} in a row.`);
    },
    table: (d) => ({ cols: ["Date", "Hours", d.top_kind === "track" ? "Top song" : "Top artist"], rows: d.days.map((r) => [r[0], r[1].toFixed(2), r[2] || "—"]) }),
  };

  // ---------------------------------------------------------------- radar (shared by 4 and 6)

  function drawRadar(el, d, w, names) {
    const { C } = V();
    const n = d.axes.length;
    if (n < 3) return V().empty(el, "Needs at least three genres with tags.");
    const colors = [C.a, C.b];
    const size = Math.min(w, 440), R = size / 2 - 64;
    const max = Math.max(0.01, d3.max(d.webs.flatMap((x) => x.values)));
    const step = max > 0.2 ? 0.05 : max > 0.08 ? 0.02 : 0.01;
    const top = Math.ceil(max / step) * step;
    const r = d3.scaleLinear([0, top], [0, R]);
    const ang = (i) => (i / n) * 2 * Math.PI - Math.PI / 2;
    const pt = (i, v) => [Math.cos(ang(i)) * r(v), Math.sin(ang(i)) * r(v)];
    const svg = d3.select(el).append("svg").attr("width", w).attr("height", size)
      .attr("role", "img").attr("aria-label", "Radar chart of genre shares; use the table view for the numbers.");
    const g = svg.append("g").attr("transform", `translate(${w / 2},${size / 2})`);
    const rings = d3.range(1, 5).map((k) => (top * k) / 4);
    for (const v of rings) {
      g.append("path").attr("fill", "none").attr("stroke", C.line)
        .attr("d", d3.line()(d3.range(n + 1).map((i) => pt(i % n, v))));
    }
    g.append("text").attr("x", 4).attr("y", -r(top) - 4).attr("fill", C.muted).attr("font-size", 10).text(pct(top));
    g.append("text").attr("x", 4).attr("y", -r(top / 2) - 4).attr("fill", C.muted).attr("font-size", 10).text(pct(top / 2));
    d.axes.forEach((a, i) => {
      const [x2, y2] = pt(i, top);
      g.append("line").attr("x2", x2).attr("y2", y2).attr("stroke", C.line);
      const [lx, ly] = pt(i, top * 1.13);
      g.append("text").attr("x", lx).attr("y", ly + 4).attr("fill", C.ink).attr("font-size", 12).attr("font-weight", 600)
        .attr("text-anchor", Math.abs(lx) < 8 ? "middle" : lx > 0 ? "start" : "end").text(V().trunc(a, 16));
    });
    d.webs.forEach((web, k) => {
      const pts = web.values.map((v, i) => pt(i, v));
      g.append("path").attr("d", d3.line()([...pts, pts[0]])).attr("fill", colors[k]).attr("fill-opacity", 0.16)
        .attr("stroke", colors[k]).attr("stroke-width", 2).attr("stroke-linejoin", "round");
    });
    d.webs.forEach((web, k) => web.values.forEach((v, i) => {
      const [x, y] = pt(i, v);
      g.append("circle").attr("cx", x).attr("cy", y).attr("r", 4).attr("fill", colors[k]).attr("stroke", C.surface).attr("stroke-width", 2);
    }));
    // Hover anywhere near an axis: both webs' values on it.
    svg.on("mousemove", (e) => {
      const [mx, my] = d3.pointer(e, g.node());
      if (Math.hypot(mx, my) > R + 30) return V().untip();
      let a = Math.atan2(my, mx) + Math.PI / 2;
      if (a < 0) a += 2 * Math.PI;
      const i = Math.round(a / (2 * Math.PI / n)) % n;
      V().tip(e, `<div class="t">${esc(d.axes[i])}</div>${d.webs.map((web, k) =>
        `<div class="s"><span class="sw" style="background:${colors[k]}"></span>${esc(names[k])}: <strong>${pct(web.values[i])}</strong></div>`).join("")}`);
    }).on("mouseleave", V().untip);
    legend(el, d.webs.map((web, k) => [colors[k], names[k]]));
  }

  function mover(d, names) {
    if (d.webs.length < 2) return "";
    const diffs = d.axes.map((a, i) => [a, d.webs[1].values[i] - d.webs[0].values[i], i]);
    const up = diffs.reduce((x, y) => (y[1] > x[1] ? y : x));
    const down = diffs.reduce((x, y) => (y[1] < x[1] ? y : x));
    const part = (t, word) => `<strong>${esc(t[0])}</strong> ${word} from ${pct(d.webs[0].values[t[2]])} (${esc(names[0])}) to ${pct(d.webs[1].values[t[2]])} (${esc(names[1])})`;
    return `${up[1] > 0.005 ? part(up, "rose") : ""}${up[1] > 0.005 && down[1] < -0.005 ? "; " : ""}${down[1] < -0.005 ? part(down, "fell") : ""}.`;
  }

  // ---------------------------------------------------------------- 4. then vs. now

  function periodName(p) {
    const yr = p.preset && /^\d{4}$/.test(p.preset);
    return yr ? p.preset : `${niceDate(p.start, { month: "short", year: "numeric" })} – ${niceDate(p.end, { month: "short", year: "numeric" })}`;
  }

  const compare = {
    id: "compare", history: true, title: "Then vs. now",
    blurb: "Your genre mix in two periods, as two webs. Each point is that genre's share of the period's listening.",
    defaults() {
      const s = L.radar, ys = L.meta.years;
      if (s.a) return;
      const [pa, pb] = ys.length >= 2 ? [String(ys.at(-2)), String(ys.at(-1))] : ["all", "6m"];
      s.a = { preset: pa }; s.b = { preset: pb };
      [s.a.start, s.a.end] = presetRange(pa); [s.b.start, s.b.end] = presetRange(pb);
    },
    controls(el) {
      const s = L.radar;
      for (const [k, p] of [["a", s.a], ["b", s.b]]) {
        const row = document.createElement("div");
        row.className = "period-row";
        row.innerHTML = `<span class="sw big" style="background:${k === "a" ? V().C.a : V().C.b}"></span>`;
        el.append(row);
        rangeControls(row, `r${k}`, p, () => refresh(compare), true);
      }
    },
    load: () => get("radar", { a_start: L.radar.a.start, a_end: L.radar.a.end, b_start: L.radar.b.start, b_end: L.radar.b.end }),
    render(el, d, w) {
      const names = [periodName(L.radar.a), periodName(L.radar.b)];
      if (!d.webs[0].hours || !d.webs[1].hours) return V().empty(el, "One of the periods has no listening.");
      drawRadar(el, d, w, names);
      V().note(el, `${mover(d, names)} These ${d.axes.length} genres cover ${pct(d.webs[0].covered)} and ${pct(d.webs[1].covered)} of the two periods' tagged listening.`);
    },
    table: (d) => ({ cols: ["Genre", periodName(L.radar.a), periodName(L.radar.b)], rows: d.axes.map((a, i) => [a, pct(d.webs[0].values[i]), pct(d.webs[1].values[i])]) }),
  };

  // ---------------------------------------------------------------- 5. artist web (chord)

  const web = {
    id: "chord", wide: true, title: "Artist web",
    blurb: "Your top artists, joined when their genres overlap. A thicker ribbon means more of the two artists' sound is shared; its color is the genre they share most.",
    controls(el) {
      const d = L.data.chord;
      if (!d) return;
      const tally = d3.rollups(d.links, (v) => d3.sum(v, (l) => l.w), (l) => l.genre).sort((a, b) => b[1] - a[1]).slice(0, 10);
      el.innerHTML = chips("genre", [["", "All genres"], ...tally.map(([g]) => [g, g])], L.chord.genre || "");
      on(el, "[data-genre]", (b) => { L.chord.genre = b.dataset.genre || null; draw(web); });
    },
    load: () => get("chord"),
    loaded: () => true,                                  // controls are built from the data
    render(el, d, w) {
      const { C } = V();
      const used = [...new Set(d.links.flatMap((l) => [l.s, l.t]))].sort((a, b) => a - b);
      if (used.length < 2) return V().empty(el, "Not enough tagged artists that share a genre yet. Run the Tags step on Home.");
      const at = new Map(used.map((k, i) => [k, i]));
      const arts = used.map((k) => d.artists[k]);
      const M = arts.map(() => new Array(arts.length).fill(0));
      const linkOf = new Map();
      for (const l of d.links) {
        const i = at.get(l.s), j = at.get(l.t);
        M[i][j] = M[j][i] = l.w;
        linkOf.set(`${Math.min(i, j)},${Math.max(i, j)}`, l);
      }
      const size = Math.min(w, 620), R = Math.max(60, size / 2 - 110);
      const svg = d3.select(el).append("svg").attr("width", w).attr("height", size)
        .attr("role", "img").attr("aria-label", "Chord diagram of artists sharing genres; use the table view for the pairs.");
      const g = svg.append("g").attr("transform", `translate(${w / 2},${size / 2})`);
      const chords = d3.chord().padAngle(0.035).sortSubgroups(d3.descending)(M);
      const lk = (c) => linkOf.get(`${Math.min(c.source.index, c.target.index)},${Math.max(c.source.index, c.target.index)}`);
      const sel = L.chord.genre;
      const hasG = (l) => !sel || l.shared.some(([x]) => x === sel);
      const ribbons = g.append("g").selectAll("path").data(chords).join("path")
        .attr("d", d3.ribbon().radius(R - 2))
        .attr("fill", (c) => genreColor(sel && hasG(lk(c)) ? sel : lk(c).genre))
        .attr("fill-opacity", (c) => (hasG(lk(c)) ? 0.62 : 0.06))
        .attr("stroke", C.surface).attr("stroke-width", 0.5)
        .on("mousemove", (e, c) => {
          const l = lk(c);
          V().tip(e, `<div class="t">${esc(arts[c.source.index].name)} ↔ ${esc(arts[c.target.index].name)}</div>
            <div class="s">${pct(l.w)} genre overlap</div><div class="s">Shared: ${l.shared.map(([x, v]) => `${esc(x)} ${pct(v)}`).join(", ")}</div>`);
        })
        .on("mouseleave", V().untip);
      const groupColor = (i) => genreColor(arts[i].genres[0][0]);
      const arc = d3.arc().innerRadius(R).outerRadius(R + 12).cornerRadius(2);
      g.append("g").selectAll("path").data(chords.groups).join("path").attr("d", arc).attr("fill", (a) => groupColor(a.index))
        .attr("stroke", C.surface).attr("stroke-width", 1)
        .on("mousemove", (e, a) => {
          ribbons.attr("fill-opacity", (c) => (c.source.index === a.index || c.target.index === a.index ? 0.8 : 0.04));
          const ar = arts[a.index];
          V().tip(e, `<div class="t">${esc(ar.name)}</div><div class="s">${d.unit === "hours" ? `${hrs(ar.value)} played` : `${ar.value} liked songs`}</div>
            <div class="s">${ar.genres.slice(0, 3).map(([x, v]) => `${esc(x)} ${pct(v)}`).join(", ")}</div>`);
        })
        .on("mouseleave", () => { ribbons.attr("fill-opacity", (c) => (hasG(lk(c)) ? 0.62 : 0.06)); V().untip(); });
      g.append("g").selectAll("text").data(chords.groups).join("text")
        .each((a) => { a.mid = (a.startAngle + a.endAngle) / 2; })
        .attr("transform", (a) => `rotate(${(a.mid * 180) / Math.PI - 90}) translate(${R + 18})${a.mid > Math.PI ? " rotate(180)" : ""}`)
        .attr("text-anchor", (a) => (a.mid > Math.PI ? "end" : "start")).attr("dy", "0.35em")
        .attr("fill", C.ink).attr("font-size", 11).text((a) => V().trunc(arts[a.index].name, 16));
      const shown = [...new Set(d.links.map((l) => l.genre))].sort((a, b) => L.meta.genre_rank.indexOf(a) - L.meta.genre_rank.indexOf(b));
      const named = shown.filter((x) => genreColor(x) !== OTHER);
      legend(el, [...named.map((x) => [genreColor(x), x]), ...(shown.length > named.length ? [[OTHER, "Other genres"]] : [])]);
      const best = d.links.reduce((a, b) => (b.w > a.w ? b : a));
      V().note(el, `Closest pair: <strong>${esc(d.artists[best.s].name)}</strong> and <strong>${esc(d.artists[best.t].name)}</strong>, ${pct(best.w)} alike, mostly ${esc(best.genre)}.
        ${d.lonely.length ? `${d.lonely.length} of your top ${d.artists.length} artists share no genre with the rest (${esc(d.lonely.slice(0, 3).join(", "))}${d.lonely.length > 3 ? "…" : ""}).` : ""}
        Ranked by ${d.unit === "hours" ? "hours played" : "liked songs"}.`);
    },
    table: (d) => ({ cols: ["Artist", "Artist", "Overlap", "Shared genres"], rows: [...d.links].sort((a, b) => b.w - a.w).map((l) => [d.artists[l.s].name, d.artists[l.t].name, pct(l.w), l.shared.map((x) => x[0]).join(", ")]) }),
  };

  // ---------------------------------------------------------------- 6. overall taste

  const overall = {
    id: "overall", title: "Your taste, overall",
    blurb: "Your genre mix across everything: what you actually play (hours) next to what you've liked (songs).",
    load: () => get("overall"),
    render(el, d, w) {
      const names = d.webs.map((x) => x.label);
      drawRadar(el, d, w, names);
      const lead = d3.maxIndex(d.webs[0].values);
      let gap = "Import your listening history to compare it with what you play.";
      if (d.webs.length > 1) {
        const [play, like] = d.webs.map((x) => x.values);
        const i = d3.maxIndex(d.axes, (_, k) => like[k] - play[k]), j = d3.minIndex(d.axes, (_, k) => like[k] - play[k]);
        gap = `You like more <strong>${esc(d.axes[i])}</strong> than you play (${pct(like[i])} of liked songs, ${pct(play[i])} of hours);
          <strong>${esc(d.axes[j])}</strong> is the reverse (${pct(like[j])} vs. ${pct(play[j])}).`;
      }
      V().note(el, `Your center of gravity is <strong>${esc(d.axes[lead])}</strong> (${pct(d.webs[0].values[lead])} of ${d.webs[0].unit === "hours" ? "your listening" : "your liked songs"}). ${gap}`);
    },
    table: (d) => ({ cols: ["Genre", ...d.webs.map((x) => x.label)], rows: d.axes.map((a, i) => [a, ...d.webs.map((x) => pct(x.values[i]))]) }),
  };

  const VIZ = [timeline, genreMix, compare, calendar, web, overall];

  // ---------------------------------------------------------------- cards

  function cardBody(v) { return $(`[data-lviz="${v.id}"] .viz-body`); }
  const modalFor = () => $("#modal").hidden ? null : $("#modal-content").dataset.lviz;

  async function draw(v) {
    const targets = [cardBody(v)];
    if (modalFor() === v.id) targets.push($("#modal-content"));
    for (const body of targets) if (body) await drawInto(body, v);
  }
  async function refresh(v) { delete L.data[v.id]; await draw(v); }

  async function drawInto(body, v) {
    V().untip();
    if (v.history && !L.meta.has_history) {
      body.innerHTML = `<div class="viz-empty">Needs your listening history. Import it above.</div>`;
      return;
    }
    v.defaults?.();
    const tok = (L.tok[v.id] = (L.tok[v.id] || 0) + 1);
    let d = L.data[v.id];
    if (!d) {
      body.innerHTML = `<div class="viz-empty">Loading…</div>`;
      try { d = await v.load(); } catch (e) { body.innerHTML = `<div class="viz-empty">${esc(e.message)}</div>`; return; }
      if (tok !== L.tok[v.id]) return;                        // a newer draw started meanwhile
      L.data[v.id] = d;
      v.loaded?.(d);                                          // e.g. default lines picked by the server
    }
    body.innerHTML = "";
    const ctr = document.createElement("div");
    ctr.className = "lcontrols";
    body.append(ctr);
    v.controls?.(ctr);
    if (!ctr.childElementCount) ctr.remove();
    const host = document.createElement("div");
    body.append(host);
    if (L.tables.has(v.id)) host.innerHTML = V().tableHtml(v.table(d));
    else v.render(host, d, Math.max(260, Math.floor(body.clientWidth)));
  }

  function renderHead() {
    const m = L.meta, s = App.session;
    const stat = (lbl, value, sub = "") => `<div class="stat"><div class="label">${lbl}</div><div class="value">${value}</div>${sub ? `<div class="sub">${sub}</div>` : ""}</div>`;
    $("#listen-stats").innerHTML = m.has_history ? [
      stat("Hours listened", V().fmt(Math.round(m.hours)), `${V().fmt(m.plays)} plays`),
      stat("Per day", hrs(m.hours / Math.max(1, (parseDay(m.last) - parseDay(m.first)) / DAY + 1)), "on average"),
      stat("History", niceDate(m.first, { month: "short", year: "numeric" }), `to ${niceDate(m.last, { month: "short", year: "numeric" })}`),
      stat("With a genre", pct(m.tagged), "of hours have Last.fm tags"),
    ].join("") : "";
    $("#listen-title").textContent = s.mode === "demo" ? "The demo's listening" : "Your listening";

    const imp = $("#import");
    if (s.mode === "demo") { imp.innerHTML = `<p class="muted">The demo library comes with a synthetic streaming history: 4¾ years of plays with genres that come and go.</p>`; return; }
    const open = !m.has_history || L.importOpen;
    imp.innerHTML = `${m.has_history ? `<div class="import-bar"><span>${V().fmt(m.plays)} plays imported${m.tagged < 0.9 ? ` · ${pct(1 - m.tagged)} of hours still need a genre` : ""}.</span>
        ${m.tagged < 0.9 ? `<button class="btn small primary" data-run-tags>Fetch genres</button>` : ""}
        <button class="btn small outline" data-toggle-import>${L.importOpen ? "Close" : "Import more"}</button></div>` : ""}
      ${open ? `<div class="import-card">
        <h3>${m.has_history ? "Add or replace history" : "Import your listening history"}</h3>
        <p class="muted">Spotify only shares your last 50 plays through its API, so these charts read your data export instead.</p>
        <ol>
          <li>Open <a href="https://www.spotify.com/account/privacy/" target="_blank" rel="noopener">spotify.com/account/privacy</a> and request <strong>Extended streaming history</strong> (every play, ever; can take up to 30 days). <strong>Account data</strong> is faster (about 5 days) but only covers the last year.</li>
          <li>When Spotify emails you, download the .zip.</li>
          <li>Drop it here, as is. Only the streaming-history files are read, and nothing leaves this machine.</li>
        </ol>
        <label class="dropzone" id="dropzone"><input type="file" id="history-file" accept=".zip,.json" multiple hidden>
          <strong>Drop the .zip here</strong><span class="muted">or click to choose it (the JSON files inside work too)</span></label>
        ${m.has_history ? `<label class="check"><input type="checkbox" id="history-replace"> Replace the history already imported (otherwise it's merged)</label>` : ""}
      </div>` : ""}`;
  }

  async function upload(files) {
    if (!files.length) return;
    const fd = new FormData();
    for (const f of files) fd.append("files", f);
    if ($("#history-replace")?.checked) fd.append("replace", "1");
    $("#dropzone").classList.add("busy");
    $("#dropzone strong").textContent = "Reading your history…";
    const r = await fetch("/api/history", { method: "POST", body: fd });
    const j = await r.json().catch(() => ({ ok: false, error: `HTTP ${r.status}` }));
    if (!j.ok) { renderHead(); throw new Error(j.error); }
    toast(`Imported ${j.summary}`);
    L.importOpen = false;
    await show(true);
  }

  async function show(force = false) {
    const key = `${App.session.mode}:${App.session.done.pull}`;
    if (force || L.key !== key) {
      L.key = key;
      L.data = {};
      $("#listen-grid").innerHTML = `<div class="viz-empty">Loading your listening…</div>`;
    }
    let meta;
    try { meta = await get("meta"); } catch (e) {
      const stale = /HTTP 404/.test(e.message);           // server started before this view existed
      $("#listen-grid").innerHTML = `<div class="viz-empty">${stale
        ? "The app server is out of date. Restart <code>python -m src.app</code> and reload this page."
        : esc(e.message)}</div>`;
      L.key = null;
      return;
    }
    if (JSON.stringify(meta) !== JSON.stringify(L.meta)) L.data = {};     // new history or tags
    L.meta = meta;
    renderHead();
    $("#listen-grid").innerHTML = VIZ.map((v) => `<article class="viz ${v.wide ? "wide" : ""}" data-lviz="${v.id}">
      <div class="viz-head"><div><h3>${esc(v.title)}</h3><p>${esc(v.blurb)}</p></div>
        <div class="tools">
          <button class="btn" data-ltable="${v.id}" aria-pressed="${L.tables.has(v.id)}" title="Show as table" aria-label="Show ${esc(v.title)} as a table"><svg><use href="#i-table"/></svg></button>
          <button class="btn" data-lexpand="${v.id}" title="Expand" aria-label="Expand ${esc(v.title)}"><svg><use href="#i-expand"/></svg></button>
        </div></div>
      <div class="viz-body"></div></article>`).join("");
    await Promise.all(VIZ.map((v) => drawInto(cardBody(v), v)));
  }

  // ---------------------------------------------------------------- events

  document.addEventListener("click", guard(async (e) => {
    if (App.view !== "listening" && !modalFor()) return;
    const tb = e.target.closest("[data-ltable]");
    if (tb) {
      const id = tb.dataset.ltable;
      L.tables.has(id) ? L.tables.delete(id) : L.tables.add(id);
      tb.setAttribute("aria-pressed", String(L.tables.has(id)));
      return draw(VIZ.find((v) => v.id === id));
    }
    const ex = e.target.closest("[data-lexpand]");
    if (ex) {
      const v = VIZ.find((x) => x.id === ex.dataset.lexpand);
      $("#modal-title").textContent = v.title;
      $("#modal").hidden = false;
      const body = $("#modal-content");
      body.className = "viz-body";
      body.dataset.lviz = v.id;
      return drawInto(body, v);
    }
    if (e.target.closest("#modal-close") || e.target.id === "modal") delete $("#modal-content").dataset.lviz;
    if (e.target.closest("[data-toggle-import]")) { L.importOpen = !L.importOpen; return renderHead(); }
    if (e.target.closest("[data-run-tags]")) {
      await api("/api/run", { steps: ["tags"] });
      await refreshSession();
      toast("Fetching genres from Last.fm. Follow it on Home; reopen Listening when it's done.");
    }
  }));
  document.addEventListener("change", guard(async (e) => { if (e.target.id === "history-file") await upload([...e.target.files]); }));
  for (const ev of ["dragover", "dragleave", "drop"]) {
    document.addEventListener(ev, guard(async (e) => {
      const z = e.target.closest?.("#dropzone");
      if (!z) return;
      e.preventDefault();
      z.classList.toggle("over", ev === "dragover");
      if (ev === "drop") await upload([...e.dataTransfer.files]);
    }));
  }
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") delete $("#modal-content").dataset.lviz; });

  let resizeTimer, lastW = innerWidth;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (App.view === "listening" && L.meta && Math.abs(innerWidth - lastW) > 20) { lastW = innerWidth; VIZ.forEach((v) => draw(v)); }
    }, 200);
  });

  App.hooks.view.push((name) => { if (name === "listening") guard(show)(); });
})();

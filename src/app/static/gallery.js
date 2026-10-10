// Taste gallery: six views of one library, drawn with d3 from /api/insights.
// Colors: green = the highlighted thing, violet = a second highlight, grey = everything else;
// magnitudes use one green ramp, similarity uses blue (unlike) <-> grey <-> green (alike).
"use strict";

(() => {
  const C = {
    a: "#1aa34a", b: "#9085e9", rest: "#4d4d4d", surface: "#181818", empty: "#242424",
    ink: "#ffffff", muted: "#b3b3b3", line: "#2a2a2a",
  };
  const seq = (t) => d3.interpolateRgb("#1d3526", "#1ed760")(Math.max(0, Math.min(1, t)));
  const div = d3.piecewise(d3.interpolateRgb, ["#3987e5", "#3e3e3e", "#1ed760"]);
  const DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
  const hourLabel = (h) => `${h % 12 || 12} ${h < 12 ? "AM" : "PM"}`;
  const trunc = (s, n) => (s.length > n ? s.slice(0, n - 1) + "…" : s);
  const fmt = d3.format(",");
  const inkOn = (hex) => (d3.lab(hex).l > 62 ? "#000" : "#fff");

  const G = { data: null, key: null, pick: [0], tables: new Set() };

  // ---------------------------------------------------------------- tooltip

  const tipEl = () => $("#tip");
  function tip(e, html) {
    const t = tipEl();
    t.innerHTML = html;
    t.hidden = false;
    const r = t.getBoundingClientRect();
    let x = e.clientX + 14, y = e.clientY + 14;
    if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 14;
    if (y + r.height > innerHeight - 8) y = e.clientY - r.height - 14;
    t.style.left = `${x}px`;
    t.style.top = `${y}px`;
  }
  const untip = () => (tipEl().hidden = true);

  // ---------------------------------------------------------------- 1. taste galaxy

  function groupLabel(g) { return g.kind === "new" ? `New: ${g.name}` : g.name; }

  const galaxy = {
    id: "galaxy", wide: true, title: "Taste galaxy",
    blurb: "Every song as a star. Songs that sound alike sit together. Pick a playlist to light up its constellation, and a second to compare.",
    render(el, d, w) {
      const gx = d.galaxy;
      if (!gx) return empty(el, "Needs the sorter's vector space; run Sort again.");
      const groups = d.groups;
      G.pick = G.pick.filter((i) => i < groups.length);
      const picker = document.createElement("div");
      picker.className = "picker";
      picker.innerHTML = groups.map((g, i) => {
        const k = G.pick.indexOf(i);
        return `<button class="chip ${k === 0 ? "a" : k === 1 ? "b" : ""} ${g.kind === "new" ? "new" : ""}" data-g="${i}">${esc(trunc(g.name, 28))} · ${fmt(g.size)}</button>`;
      }).join("");
      picker.addEventListener("click", (e) => {
        const b = e.target.closest("[data-g]");
        if (!b) return;
        const i = +b.dataset.g;
        G.pick = G.pick.includes(i) ? G.pick.filter((x) => x !== i) : [...G.pick, i].slice(-2);
        drawAll();
      });
      el.append(picker);

      const h = Math.round(Math.min(560, Math.max(320, w * 0.55)));
      const dpr = window.devicePixelRatio || 1;
      const canvas = document.createElement("canvas");
      canvas.width = w * dpr; canvas.height = h * dpr;
      canvas.style.width = `${w}px`; canvas.style.height = `${h}px`;
      canvas.setAttribute("role", "img");
      canvas.setAttribute("aria-label", "Scatter of songs by similarity; use the table view for the numbers.");
      el.append(canvas);
      const ctx = canvas.getContext("2d");
      ctx.scale(dpr, dpr);
      const m = 14;
      const x = d3.scaleLinear([-1.05, 1.05], [m, w - m]);
      const y = d3.scaleLinear([-1.05, 1.05], [h - m, m]);
      const [A, B] = G.pick;
      const role = (p) => (p[4].includes(A) ? 0 : p[4].includes(B) ? 1 : 2);
      const pts = gx.points.map((p) => ({ p, sx: x(p[0]), sy: y(p[1]), r: role(p) }));

      for (const q of pts) if (q.r === 2) dot(ctx, q, C.rest, 1.8, 0.6, false);
      for (const q of pts) if (q.r === 1) dot(ctx, q, C.b, 3.5, 1, true);
      for (const q of pts) if (q.r === 0) dot(ctx, q, C.a, 3.5, 1, true);

      const tree = d3.quadtree(pts, (q) => q.sx, (q) => q.sy);
      canvas.addEventListener("mousemove", (e) => {
        const r = canvas.getBoundingClientRect();
        const q = tree.find(e.clientX - r.left, e.clientY - r.top, 10);
        if (!q) return untip();
        const names = q.p[4].slice(0, 3).map((i) => esc(groupLabel(groups[i])));
        tip(e, `<div class="t">${esc(q.p[2])}</div><div class="s">${esc(q.p[3])}</div>
          ${names.length ? `<div class="s">In: ${names.join(", ")}${q.p[4].length > 3 ? "…" : ""}</div>` : ""}`);
      });
      canvas.addEventListener("mouseleave", untip);

      const leg = document.createElement("div");
      leg.className = "legend";
      leg.innerHTML = [A, B].map((i, k) => i == null ? "" :
        `<span><span class="sw" style="background:${k ? C.b : C.a}"></span>${esc(groupLabel(groups[i]))}</span>`).join("")
        + `<span><span class="sw" style="background:${C.rest}"></span>Everything else</span>`;
      el.append(leg);
      note(el, `Axes are the two strongest directions of variation in the sorter's <strong>${esc(gx.space)}</strong> space
        (${pct(gx.explained[0])} and ${pct(gx.explained[1])} of it), so the map is a squashed shadow: tight clusters are real, gaps can hide.`);
    },
    table: (d) => ({ cols: ["Group", "Kind", "Songs"], rows: d.groups.map((g) => [g.name, g.kind, g.size]) }),
  };

  function dot(ctx, q, color, r, alpha, ring) {
    ctx.globalAlpha = alpha;
    ctx.beginPath();
    ctx.arc(q.sx, q.sy, r, 0, 2 * Math.PI);
    if (ring) { ctx.lineWidth = 1.5; ctx.strokeStyle = C.surface; ctx.stroke(); }
    ctx.fillStyle = color;
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  // ---------------------------------------------------------------- 2. time machine

  const timeMachine = {
    id: "eras", title: "Time machine",
    blurb: "When each song came out against when you liked it. The white line follows the typical release year over time.",
    render(el, d, w) {
      const rows = d.eras.map(([yr, at, name, artist]) => ({ yr, at: at ? new Date(at) : null, name, artist }));
      if (!rows.length) return empty(el, "No release dates in this library.");
      const dated = rows.filter((r) => r.at && !isNaN(r.at));
      const h = 320, m = { t: 12, r: 64, b: 28, l: 44 };
      const svg = d3.select(el).append("svg").attr("width", w).attr("height", h)
        .attr("role", "img").attr("aria-label", "Release year against date liked");
      const yrs = d3.extent(rows, (r) => r.yr);
      const y = d3.scaleLinear([yrs[0] - 1, yrs[1] + 1], [h - m.b, m.t]).nice();
      svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`)
        .call(d3.axisLeft(y).ticks(6).tickFormat(d3.format("d")).tickSize(-(w - m.l - m.r)))
        .call((g) => g.select(".domain").remove()).call((g) => g.selectAll("line").attr("stroke", C.line));

      if (dated.length >= 10) {
        const x = d3.scaleTime(d3.extent(dated, (r) => r.at), [m.l, w - m.r]).nice();
        svg.append("g").attr("class", "axis").attr("transform", `translate(0,${h - m.b})`)
          .call(d3.axisBottom(x).ticks(Math.max(2, Math.floor(w / 110))).tickSizeOuter(0));
        svg.append("g").selectAll("circle").data(dated).join("circle")
          .attr("cx", (r) => x(r.at)).attr("cy", (r) => y(r.yr)).attr("r", 2.5)
          .attr("fill", C.a).attr("fill-opacity", 0.45);
        // Rolling median release year, one point per month.
        const months = d3.groups(dated, (r) => d3.timeMonth(r.at)).sort((a, b) => a[0] - b[0]);
        const med = months.map(([mo], i) => {
          const win = months.slice(Math.max(0, i - 2), i + 1).flatMap((x) => x[1]);
          return [mo, d3.median(win, (r) => r.yr)];
        });
        svg.append("path").datum(med).attr("fill", "none").attr("stroke", C.ink).attr("stroke-width", 2)
          .attr("stroke-linejoin", "round").attr("stroke-linecap", "round")
          .attr("d", d3.line((p) => x(p[0]), (p) => y(p[1])).curve(d3.curveMonotoneX));
        const last = med[med.length - 1];
        svg.append("text").attr("x", x(last[0]) + 6).attr("y", y(last[1]) + 4).attr("fill", C.muted)
          .attr("font-size", 11).text(`median ${Math.round(last[1])}`);
        const tree = d3.quadtree(dated, (r) => x(r.at), (r) => y(r.yr));
        hover(svg, (mx, my) => tree.find(mx, my, 10),
          (r) => `<div class="t">${esc(r.name)}</div><div class="s">${esc(r.artist)}</div><div class="s">Released ${r.yr} · liked ${r.at.toLocaleDateString()}</div>`);
        const gap = d3.median(dated, (r) => r.at.getFullYear() - r.yr);
        note(el, `Half your liked songs came out before <strong>${d.summary.median_year}</strong>. The typical song was
          <strong>${gap} year${gap === 1 ? "" : "s"} old</strong> when you liked it.`);
      } else {
        const byYear = d3.rollups(rows, (v) => v.length, (r) => r.yr).sort((a, b) => a[0] - b[0]);
        const x = d3.scaleBand(byYear.map((b) => b[0]), [m.l, w - m.r]).padding(0.15);
        const yc = d3.scaleLinear([0, d3.max(byYear, (b) => b[1])], [h - m.b, m.t]).nice();
        svg.selectAll("g.axis").remove();
        svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`)
          .call(d3.axisLeft(yc).ticks(5).tickSize(-(w - m.l - m.r))).call((g) => g.select(".domain").remove())
          .call((g) => g.selectAll("line").attr("stroke", C.line));
        svg.append("g").attr("class", "axis").attr("transform", `translate(0,${h - m.b})`)
          .call(d3.axisBottom(x).tickValues(x.domain().filter((yr) => yr % 10 === 0)));
        svg.append("g").selectAll("path").data(byYear).join("path").attr("fill", C.a)
          .attr("d", (b) => topRounded(x(b[0]), yc(b[1]), Math.min(24, x.bandwidth()), h - m.b - yc(b[1])))
          .on("mousemove", (e, b) => tip(e, `<div class="t">${b[0]}</div><div class="s">${songs(b[1])}</div>`)).on("mouseleave", untip);
        note(el, `Half your liked songs came out before <strong>${d.summary.median_year}</strong>. (No "liked at" dates, so this shows release years only.)`);
      }
    },
    table(d) {
      const by = d3.rollups(d.eras, (v) => v.length, (r) => 10 * Math.floor(r[0] / 10)).sort((a, b) => a[0] - b[0]);
      const n = d3.sum(by, (b) => b[1]);
      return { cols: ["Decade", "Songs", "Share"], rows: by.map(([dec, c]) => [`${dec}s`, c, pct(c / n)]) };
    },
  };

  function topRounded(x, y, w, h, r = 4) {
    r = Math.min(r, h, w / 2);
    return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
  }

  // ---------------------------------------------------------------- 3. listening clock

  function clockBins(d) {
    const bins = d3.range(7).map(() => new Array(24).fill(0));
    let n = 0;
    for (const [, at] of d.eras) {
      if (!at) continue;
      const t = new Date(at);
      if (isNaN(t)) continue;
      bins[(t.getDay() + 6) % 7][t.getHours()]++;
      n++;
    }
    return { bins, n };
  }

  const clock = {
    id: "clock", title: "Listening clock",
    blurb: "When you hit the heart. Each ring is a day of the week, each slice an hour, in your time zone.",
    render(el, d, w) {
      const { bins, n } = clockBins(d);
      if (n < 10) return empty(el, "No \"liked at\" times in this library.");
      const size = Math.min(w, 420), R = size / 2 - 26, r0 = R * 0.34, ring = (R - r0) / 7;
      const gap = 0.42, span = 2 * Math.PI - gap, slice = span / 24;
      const ang = (hh) => gap / 2 + hh * slice;
      const max = d3.max(bins.flat());
      const svg = d3.select(el).append("svg").attr("width", w).attr("height", size)
        .attr("role", "img").attr("aria-label", "Radial heatmap of likes by weekday and hour");
      const g = svg.append("g").attr("transform", `translate(${w / 2},${size / 2})`);
      const arc = d3.arc().padAngle(0.012).padRadius(R).cornerRadius(2);
      const cells = bins.flatMap((row, day) => row.map((c, hh) => ({ day, hh, c })));
      const center = g.append("g").attr("text-anchor", "middle");
      const peak = cells.reduce((a, b) => (b.c > a.c ? b : a));
      const setCenter = (cell, label) => {
        center.selectAll("*").remove();
        center.append("text").attr("y", -14).attr("fill", C.muted).attr("font-size", 12).text(label);
        center.append("text").attr("y", 8).attr("fill", C.ink).attr("font-size", 17).attr("font-weight", 800)
          .text(`${DAYS[cell.day].slice(0, 3)} · ${hourLabel(cell.hh)}`);
        center.append("text").attr("y", 26).attr("fill", C.muted).attr("font-size", 12).text(songs(cell.c));
      };
      g.selectAll("path").data(cells).join("path")
        .attr("d", (c) => arc({ innerRadius: r0 + c.day * ring + 1, outerRadius: r0 + (c.day + 1) * ring - 1, startAngle: ang(c.hh), endAngle: ang(c.hh + 1) }))
        .attr("fill", (c) => (c.c ? seq(c.c / max) : C.empty))
        .on("mousemove", (e, c) => { setCenter(c, "Hovering"); tip(e, `<div class="t">${DAYS[c.day]}s, ${hourLabel(c.hh)}</div><div class="s">${songs(c.c)} liked</div>`); })
        .on("mouseleave", () => { untip(); setCenter(peak, "Your peak"); });
      setCenter(peak, "Your peak");
      for (const hh of [0, 6, 12, 18]) {
        const a = ang(hh) - Math.PI / 2;
        g.append("text").attr("x", Math.cos(a) * (R + 14)).attr("y", Math.sin(a) * (R + 14) + 4)
          .attr("text-anchor", "middle").attr("fill", C.muted).attr("font-size", 11).text(hourLabel(hh));
      }
      DAYS.forEach((dname, day) => g.append("text").attr("x", 0).attr("y", -(r0 + (day + 0.5) * ring) + 3.5)
        .attr("text-anchor", "middle").attr("fill", C.muted).attr("font-size", Math.min(10, ring - 2)).text(dname[0]));
      legendRamp(el, "Fewer likes", "More likes");
      const weekend = d3.sum(bins.slice(5).flat()) / n;
      const night = d3.sum(bins.flatMap((row) => [...row.slice(22), ...row.slice(0, 4)])) / n;
      note(el, `<strong>${pct(weekend)}</strong> of your likes land on weekends (2 of 7 days is 29%), and
        <strong>${pct(night)}</strong> between 10 PM and 4 AM. Rings run Monday (inside) to Sunday (outside).`);
    },
    table(d) {
      const { bins } = clockBins(d);
      const rows = bins.flatMap((row, day) => row.map((c, hh) => [DAYS[day], hourLabel(hh), c])).filter((r) => r[2]);
      return { cols: ["Day", "Hour", "Songs liked"], rows: rows.sort((a, b) => b[2] - a[2]) };
    },
  };

  // ---------------------------------------------------------------- 4. artist orbit

  const orbit = {
    id: "orbit", title: "Artist orbit",
    blurb: "Your most-liked lead artists, sized by how many of their songs you've liked.",
    render(el, d, w) {
      if (!d.orbit.length) return empty(el, "No artists yet.");
      const h = Math.round(Math.min(460, Math.max(300, w * 0.8)));
      const root = d3.pack().size([w, h]).padding(3)(d3.hierarchy({ children: d.orbit }).sum((a) => a.songs));
      const max = d.orbit[0].songs;
      const svg = d3.select(el).append("svg").attr("width", w).attr("height", h)
        .attr("role", "img").attr("aria-label", "Packed bubbles of top artists");
      const node = svg.selectAll("g").data(root.leaves()).join("g").attr("transform", (n) => `translate(${n.x},${n.y})`);
      node.append("circle").attr("r", (n) => n.r).attr("fill", (n) => seq(0.15 + 0.85 * n.data.songs / max))
        .on("mousemove", (e, n) => tip(e, `<div class="t">${esc(n.data.artist)}</div><div class="s">${songs(n.data.songs)}${n.data.tag ? ` · mostly ${esc(n.data.tag)}` : ""}</div>`))
        .on("mouseleave", untip);
      node.filter((n) => n.r > 30).each(function (n) {
        const fill = seq(0.15 + 0.85 * n.data.songs / max), ink = inkOn(fill);
        const t = d3.select(this).append("text").attr("text-anchor", "middle").attr("pointer-events", "none").attr("fill", ink);
        t.append("tspan").attr("x", 0).attr("dy", "-0.1em").attr("font-weight", 700)
          .attr("font-size", Math.min(15, n.r / 3.6)).text(trunc(n.data.artist, Math.floor(n.r / 3.6)));
        t.append("tspan").attr("x", 0).attr("dy", "1.3em").attr("font-size", Math.min(12, n.r / 4.5)).attr("opacity", 0.8).text(n.data.songs);
      });
      const s = d.summary, top = d.orbit[0];
      note(el, `<strong>${esc(top.artist)}</strong> alone is ${pct(top.songs / s.liked)} of your liked songs.
        ${fmt(s.artists)} different lead artists across ${fmt(s.liked)} songs.`);
    },
    table: (d) => ({ cols: ["Artist", "Liked songs", "Top tag"], rows: d.orbit.map((a) => [a.artist, a.songs, a.tag || "—"]) }),
  };

  // ---------------------------------------------------------------- 5. genre DNA

  const dna = {
    id: "dna", wide: true, title: "Genre DNA",
    blurb: "The tags that make up each of your biggest playlists. Brighter means more of the playlist carries that tag.",
    render(el, d, w) {
      if (!d.dna) return empty(el, "Needs Last.fm tags; run the tags step.");
      const { tags, rows } = d.dna;
      const left = Math.min(180, w * 0.3), top = 86;
      const cell = Math.max(14, Math.min(40, (w - left - 8) / tags.length));
      const h = top + rows.length * cell + 4;
      const max = d3.max(rows.flatMap((r) => r.share)) || 1;
      const svg = d3.select(el).append("svg").attr("width", Math.min(w, left + cell * tags.length + 8)).attr("height", h)
        .attr("role", "img").attr("aria-label", "Heatmap of tag share per playlist");
      svg.append("g").selectAll("text").data(tags).join("text")
        .attr("transform", (t, i) => `translate(${left + i * cell + cell / 2},${top - 8}) rotate(-45)`)
        .attr("fill", C.muted).attr("font-size", 11).text((t) => trunc(t, 16));
      rows.forEach((r, ri) => {
        svg.append("text").attr("x", left - 10).attr("y", top + ri * cell + cell / 2 + 4).attr("text-anchor", "end")
          .attr("fill", C.ink).attr("font-size", 12).attr("font-weight", 600).text(trunc(r.playlist, Math.floor(left / 7)));
        svg.append("g").selectAll("rect").data(r.share.map((v, i) => ({ v, i }))).join("rect")
          .attr("x", ({ i }) => left + i * cell + 1).attr("y", top + ri * cell + 1)
          .attr("width", cell - 2).attr("height", cell - 2).attr("rx", 3)
          .attr("fill", ({ v }) => (v > 0 ? seq(v / max) : C.empty))
          .on("mousemove", (e, { v, i }) => {
            tip(e, `<div class="t">${esc(r.playlist)}</div><div class="s">${pct(v)} of its songs are tagged “${esc(tags[i])}”</div>`);
          })
          .on("mouseleave", untip);
      });
      legendRamp(el, "0%", `${pct(max)} of the playlist`);
      // The most distinctive playlist/tag pair: highest share relative to that tag's average.
      let best = null;
      tags.forEach((t, i) => {
        const avg = d3.mean(rows, (r) => r.share[i]) || 1e-9;
        rows.forEach((r) => { const lift = r.share[i] / avg; if (r.share[i] > 0.3 && (!best || lift > best.lift)) best = { r, t, lift }; });
      });
      if (best) note(el, `Most distinctive: <strong>${esc(best.r.playlist)}</strong> is ${best.lift.toFixed(1)}× more “${esc(best.t)}” than your playlists on average.`);
    },
    table: (d) => d.dna ? ({ cols: ["Playlist", ...d.dna.tags], rows: d.dna.rows.map((r) => [r.playlist, ...r.share.map(pct)]) }) : null,
  };

  // ---------------------------------------------------------------- 6. playlist kinship

  const kinship = {
    id: "kinship", title: "Playlist kinship",
    blurb: "How alike your playlists sound, compared with your average playlist. Green pairs overlap; blue pairs are opposites.",
    render(el, d, w) {
      const k = d.kinship;
      if (!k) return empty(el, "Needs at least two playlists with vectors.");
      const span = d3.max(k.sim.flatMap((row, i) => row.filter((_, j) => j !== i).map(Math.abs))) || 1;
      const color = (v) => div((v / span + 1) / 2);
      const n = k.names.length, left = Math.min(150, w * 0.32), top = 6;
      const cell = Math.max(12, Math.min(34, (w - left - 8) / n));
      const svg = d3.select(el).append("svg").attr("width", Math.min(w, left + n * cell + 8)).attr("height", top + n * cell + 4)
        .attr("role", "img").attr("aria-label", "Similarity matrix of playlists");
      k.sim.forEach((row, i) => {
        svg.append("text").attr("x", left - 8).attr("y", top + i * cell + cell / 2 + 4).attr("text-anchor", "end")
          .attr("fill", C.ink).attr("font-size", 11).attr("font-weight", 600).text(trunc(k.names[i], Math.floor(left / 6.5)));
        svg.append("g").selectAll("rect").data(row.map((v, j) => ({ v, j }))).join("rect")
          .attr("x", ({ j }) => left + j * cell + 1).attr("y", top + i * cell + 1)
          .attr("width", cell - 2).attr("height", cell - 2).attr("rx", 3)
          .attr("fill", ({ v, j }) => (i === j ? C.empty : color(v)))
          .on("mousemove", (e, { v, j }) => {
            tip(e, i === j ? `<div class="t">${esc(k.names[i])}</div>` :
              `<div class="t">${esc(k.names[i])} ↔ ${esc(k.names[j])}</div><div class="s">similarity ${v.toFixed(2)} (−1 opposite, 1 same)</div>`);
          })
          .on("mouseleave", untip);
      });
      const lg = document.createElement("div");
      lg.className = "legend";
      lg.innerHTML = `<span>Opposite (−${span.toFixed(2)})<span class="ramp" style="background:linear-gradient(90deg,${div(0)},${div(0.5)},${div(1)})"></span>alike (${span.toFixed(2)})</span><span>Rows and columns list the same playlists in the same order</span>`;
      el.append(lg);
      const pairs = kinPairs(k);
      if (pairs.length) note(el, `Closest pair: <strong>${esc(pairs[0][0])}</strong> and <strong>${esc(pairs[0][1])}</strong> (${pairs[0][2].toFixed(2)}): a merge candidate?
        Furthest apart: <strong>${esc(pairs.at(-1)[0])}</strong> and <strong>${esc(pairs.at(-1)[1])}</strong>.`);
    },
    table: (d) => d.kinship ? ({ cols: ["Playlist", "Playlist", "Similarity"], rows: kinPairs(d.kinship).map((p) => [p[0], p[1], p[2].toFixed(2)]) }) : null,
  };

  function kinPairs(k) {
    const out = [];
    k.sim.forEach((row, i) => row.forEach((v, j) => { if (j > i) out.push([k.names[i], k.names[j], v]); }));
    return out.sort((a, b) => b[2] - a[2]);
  }

  const VIZ = [galaxy, clock, timeMachine, dna, orbit, kinship];

  // ---------------------------------------------------------------- shared bits

  function empty(el, msg) { el.innerHTML = `<div class="viz-empty">${esc(msg)}</div>`; }
  function note(el, html) { const p = document.createElement("p"); p.className = "viz-note"; p.innerHTML = html; el.append(p); }
  function legendRamp(el, lo, hi) {
    const lg = document.createElement("div");
    lg.className = "legend";
    lg.innerHTML = `<span>${esc(lo)}<span class="ramp" style="background:linear-gradient(90deg,${seq(0)},${seq(0.5)},${seq(1)})"></span>${esc(hi)}</span>`;
    el.append(lg);
  }
  function hover(svg, find, html) {
    svg.on("mousemove", (e) => {
      const [mx, my] = d3.pointer(e);
      const r = find(mx, my);
      r ? tip(e, html(r)) : untip();
    }).on("mouseleave", untip);
  }
  function tableHtml(t) {
    if (!t) return `<div class="viz-empty">No data.</div>`;
    return `<div class="table-wrap"><table class="data-table"><thead><tr>${t.cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>
      <tbody>${t.rows.map((r) => `<tr>${r.map((v) => `<td class="${typeof v === "number" || /^[\d.,%−-]+$/.test(v) ? "num" : ""}">${esc(typeof v === "number" ? fmt(v) : v)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  }

  function drawInto(body, v, d) {
    body.innerHTML = "";
    untip();
    if (G.tables.has(v.id)) { body.innerHTML = tableHtml(v.table(d)); return; }
    v.render(body, d, Math.max(260, Math.floor(body.clientWidth)));
  }

  // ---------------------------------------------------------------- page

  function renderStats(d) {
    const s = d.summary;
    const stat = (label, value, sub = "") => `<div class="stat"><div class="label">${label}</div><div class="value">${value}</div>${sub ? `<div class="sub">${sub}</div>` : ""}</div>`;
    $("#stats").innerHTML = [
      stat("Liked songs", fmt(s.liked), `${fmt(Math.round(s.hours))} hours of music`),
      stat("Lead artists", fmt(s.artists), `${(1 / Math.max(s.variety, 1e-9)).toFixed(1)} songs per artist`),
      stat("Playlists", fmt(s.playlists), s.unsorted != null ? `${fmt(s.unsorted)} liked songs in none` : ""),
      s.top_decade ? stat("Favourite decade", `${s.top_decade}s`, `median year ${s.median_year}`) : "",
    ].join("");
    $("#gallery-title").textContent = d.demo ? "The demo library, mapped" : "Your music, mapped";
  }

  function drawAll() {
    const d = G.data;
    if (!d) return;
    renderStats(d);
    const host = $("#gallery");
    host.innerHTML = VIZ.map((v) => `<article class="viz ${v.wide ? "wide" : ""}" data-viz="${v.id}">
      <div class="viz-head"><div><h3>${esc(v.title)}</h3><p>${esc(v.blurb)}</p></div>
        <div class="tools">
          <button class="btn" data-table="${v.id}" aria-pressed="${G.tables.has(v.id)}" title="Show as table" aria-label="Show ${esc(v.title)} as a table"><svg><use href="#i-table"/></svg></button>
          <button class="btn" data-expand="${v.id}" title="Expand" aria-label="Expand ${esc(v.title)}"><svg><use href="#i-expand"/></svg></button>
        </div></div>
      <div class="viz-body"></div></article>`).join("");
    for (const v of VIZ) drawInto($(`[data-viz="${v.id}"] .viz-body`, host), v, d);
  }

  async function show() {
    if (!App.ready) return;
    const key = S.meta?.generated_at || "x";
    if (G.key !== key) {
      $("#gallery").innerHTML = `<div class="viz-empty">Mapping your library…</div>`;
      const r = await api("/api/insights");
      G.data = r;
      G.key = key;
      G.pick = [0];
    }
    drawAll();
  }

  function openModal(id) {
    const v = VIZ.find((x) => x.id === id);
    $("#modal-title").textContent = v.title;
    $("#modal").hidden = false;
    const body = $("#modal-content");
    body.className = "viz-body";
    drawInto(body, v, G.data);
  }
  const closeModal = () => { $("#modal").hidden = true; $("#modal-content").innerHTML = ""; untip(); };

  document.addEventListener("click", (e) => {
    const tb = e.target.closest("[data-table]");
    if (tb) {
      const id = tb.dataset.table;
      G.tables.has(id) ? G.tables.delete(id) : G.tables.add(id);
      tb.setAttribute("aria-pressed", String(G.tables.has(id)));
      drawInto(tb.closest(".viz").querySelector(".viz-body"), VIZ.find((v) => v.id === id), G.data);
      return;
    }
    const ex = e.target.closest("[data-expand]");
    if (ex) return openModal(ex.dataset.expand);
    if (e.target.closest("#modal-close") || e.target.id === "modal") closeModal();
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("#modal").hidden) closeModal(); });

  let resizeTimer, lastW = 0;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (App.view === "gallery" && G.data && Math.abs(innerWidth - lastW) > 20) { lastW = innerWidth; drawAll(); }
    }, 200);
  });

  // Shared with listening.js.
  App.viz = { C, seq, tip, untip, empty, note, legendRamp, tableHtml, trunc, fmt, topRounded };

  App.hooks.view.push((name) => { if (name === "gallery") guard(show)(); });
  App.hooks.ready.push(async () => { G.key = null; if (App.view === "gallery") await guard(show)(); });
})();

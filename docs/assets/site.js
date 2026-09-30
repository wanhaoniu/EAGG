/* EAGG project page: motion, point-cloud scenes, and interactions. */
(() => {
  "use strict";

  const doc = document.documentElement;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const finePointer = window.matchMedia("(hover: hover) and (pointer: fine)").matches;
  const saveData = Boolean(navigator.connection && navigator.connection.saveData);
  if (reduceMotion) doc.classList.add("reduce-motion");

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const easeOut = (t) => 1 - Math.pow(1 - t, 3);
  const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

  /* ---------------------------------------------------------------- data */

  const HANDS = [
    ["wsg_50", "WSG-50"],
    ["franka_panda", "Franka Panda"],
    ["sawyer", "Sawyer"],
    ["robotiq_3finger", "Robotiq 3F"],
    ["Barrett", "Barrett"],
    ["jaco_robot", "Jaco"],
    ["Allegro", "Allegro"],
    ["AbilityHand", "AbilityHand"],
    ["DexHand", "DexHand"],
    ["FreedomHand", "FreedomHand"],
    ["HumanHand", "HumanHand"],
  ];
  const OBJECTS = [
    ["003_cracker_box", "Cracker box"],
    ["004_sugar_box", "Sugar box"],
    ["005_tomato_soup_can", "Soup can"],
    ["006_mustard_bottle", "Mustard bottle"],
    ["011_banana", "Banana"],
    ["024_bowl", "Bowl"],
    ["025_mug", "Mug"],
  ];
  const nameOf = (list, key) => list.find(([k]) => k === key)[1];
  const render = (hand, object, rank) => `assets/explorer/${hand}/${object}_${rank}.webp`;

  const STOPS = [
    [58, 160, 218],
    [124, 108, 242],
    [232, 97, 92],
  ];
  function gradient(t) {
    t = clamp(t, 0, 1);
    const seg = t < 0.5 ? 0 : 1;
    const u = seg === 0 ? t / 0.5 : (t - 0.5) / 0.5;
    const a = STOPS[seg];
    const b = STOPS[seg + 1];
    return [lerp(a[0], b[0], u), lerp(a[1], b[1], u), lerp(a[2], b[2], u)];
  }

  function bytes(b64) {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  function decodeHand(entry) {
    const q = new Int8Array(bytes(entry.p).buffer);
    const node = bytes(entry.n);
    const pts = Float32Array.from(q, (v) => v / 127);
    const n = entry.nodes;
    const col = new Float32Array(pts.length);
    for (let i = 0; i < node.length; i++) col.set(gradient(n > 1 ? node[i] / (n - 1) : 0), i * 3);

    // Graph nodes sit at the centroid of the points that belong to them.
    const cen = new Float32Array(n * 3);
    const count = new Uint16Array(n);
    for (let i = 0; i < node.length; i++) {
      const k = node[i];
      cen[k * 3] += pts[i * 3];
      cen[k * 3 + 1] += pts[i * 3 + 1];
      cen[k * 3 + 2] += pts[i * 3 + 2];
      count[k]++;
    }
    const known = new Uint8Array(n);
    for (let k = 0; k < n; k++) {
      if (!count[k]) continue;
      cen[k * 3] /= count[k];
      cen[k * 3 + 1] /= count[k];
      cen[k * 3 + 2] /= count[k];
      known[k] = 1;
    }
    // Nodes without points (virtual links) take the mean of their placed neighbours.
    for (let pass = 0; pass < 3; pass++) {
      for (let k = 0; k < n; k++) {
        if (known[k]) continue;
        let sx = 0, sy = 0, sz = 0, m = 0;
        for (const [a, b] of entry.edges) {
          const j = a === k ? b : b === k ? a : -1;
          if (j < 0 || !known[j]) continue;
          sx += cen[j * 3];
          sy += cen[j * 3 + 1];
          sz += cen[j * 3 + 2];
          m++;
        }
        if (m) {
          cen[k * 3] = sx / m;
          cen[k * 3 + 1] = sy / m;
          cen[k * 3 + 2] = sz / m;
          known[k] = 2;
        }
      }
    }
    const nodeCol = [];
    for (let k = 0; k < n; k++) nodeCol.push(gradient(n > 1 ? k / (n - 1) : 0));
    return { pts, col, nodes: n, edges: entry.edges, cen, known, nodeCol };
  }

  function decodeObject(entry) {
    const q = new Int8Array(bytes(entry.p).buffer);
    const pts = Float32Array.from(q, (v) => v / 127);
    const col = new Float32Array(pts.length);
    for (let i = 0; i < pts.length / 3; i++) {
      const t = clamp((pts[i * 3 + 1] + 1) / 2, 0, 1);
      col[i * 3] = lerp(170, 236, t);
      col[i * 3 + 1] = lerp(118, 190, t);
      col[i * 3 + 2] = lerp(64, 128, t);
    }
    return { pts, col, nodes: 0, edges: [] };
  }

  const cloudsReady = fetch("assets/clouds.json")
    .then((r) => r.json())
    .then((json) => {
      const hands = {};
      const objects = {};
      for (const [k, v] of Object.entries(json.hands)) hands[k] = decodeHand(v);
      for (const [k, v] of Object.entries(json.objects)) objects[k] = decodeObject(v);
      return { hands, objects };
    })
    .catch(() => null);

  /* -------------------------------------------------------- scroll system */

  const scrollHandlers = [];
  let lastY = window.scrollY;
  let scrollQueued = false;
  function runScroll() {
    scrollQueued = false;
    const y = window.scrollY;
    for (const fn of scrollHandlers) fn(y);
    lastY = y;
  }
  const queueScroll = () => {
    if (!scrollQueued) {
      scrollQueued = true;
      requestAnimationFrame(runScroll);
    }
  };
  window.addEventListener("scroll", queueScroll, { passive: true });
  window.addEventListener("resize", queueScroll);

  const header = $("[data-header]");
  const progress = $(".progress span");
  scrollHandlers.push((y) => {
    header.classList.toggle("is-scrolled", y > 24);
    if (y > lastY + 6 && y > 480 && !header.matches(":focus-within")) header.classList.add("is-hidden");
    else if (y < lastY - 6 || y <= 480) header.classList.remove("is-hidden");
    const max = doc.scrollHeight - window.innerHeight;
    progress.style.setProperty("--progress", max > 0 ? (y / max).toFixed(4) : "0");
  });
  header.addEventListener("focusin", () => header.classList.remove("is-hidden"));

  // Section spy with a sliding pill under the active link.
  const nav = $(".nav");
  const navLinks = $$("[data-nav] a");
  const indicator = $("[data-nav-indicator]");
  let currentLink = null;
  function placeIndicator() {
    if (!currentLink || !nav.offsetParent) {
      indicator.style.opacity = "0";
      return;
    }
    const nr = nav.getBoundingClientRect();
    const lr = currentLink.getBoundingClientRect();
    indicator.style.width = `${lr.width}px`;
    indicator.style.transform = `translateX(${lr.left - nr.left}px)`;
    indicator.style.opacity = "1";
  }
  if ("IntersectionObserver" in window) {
    const inView = new Set();
    const spy = new IntersectionObserver(
      (entries) => {
        for (const e of entries) e.isIntersecting ? inView.add(e.target.id) : inView.delete(e.target.id);
        currentLink = navLinks.find((l) => inView.has(l.hash.slice(1))) || null;
        navLinks.forEach((l) => (l === currentLink ? l.setAttribute("aria-current", "true") : l.removeAttribute("aria-current")));
        placeIndicator();
      },
      { rootMargin: "-45% 0px -54% 0px" }
    );
    navLinks.forEach((l) => {
      const target = document.getElementById(l.hash.slice(1));
      if (target) spy.observe(target);
    });
    window.addEventListener("resize", placeIndicator);
  }

  /* ------------------------------------------------------------- reveals */

  const onceVisible = (els, fn, options = { rootMargin: "0px 0px -8% 0px", threshold: 0.08 }) => {
    if (reduceMotion || !("IntersectionObserver" in window)) {
      els.forEach(fn);
      return;
    }
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        fn(e.target);
        io.unobserve(e.target);
      }
    }, options);
    els.forEach((el) => io.observe(el));
  };

  onceVisible($$("[data-reveal]"), (el) => el.classList.add("is-in"));
  onceVisible($$("[data-bars], [data-gauges]"), (el) => el.classList.add("is-in"), { threshold: 0.3 });

  function countUp(el) {
    const target = parseFloat(el.dataset.count);
    const decimals = Number(el.dataset.decimals || 0);
    const t0 = performance.now();
    const tick = (now) => {
      const p = clamp((now - t0) / 1600, 0, 1);
      el.textContent = (target * easeOut(p)).toFixed(decimals);
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
  if (!reduceMotion) {
    const counters = $$("[data-count]");
    counters.forEach((el) => (el.textContent = (0).toFixed(Number(el.dataset.decimals || 0))));
    onceVisible(counters, countUp, { threshold: 0.6 });
  }

  // Statement: words light up as the paragraph scrolls through the viewport.
  const statement = $("[data-words]");
  if (statement) {
    const words = [];
    const wrap = (text, extra) => {
      const frag = document.createDocumentFragment();
      for (const part of text.split(/(\s+)/)) {
        if (!part) continue;
        if (/^\s+$/.test(part)) {
          frag.appendChild(document.createTextNode(" "));
          continue;
        }
        const span = document.createElement("span");
        span.className = extra ? `w ${extra}` : "w";
        span.textContent = part;
        words.push(span);
        frag.appendChild(span);
      }
      return frag;
    };
    for (const node of [...statement.childNodes]) {
      if (node.nodeType === Node.TEXT_NODE) {
        statement.replaceChild(wrap(node.textContent), node);
      } else if (node.nodeType === Node.ELEMENT_NODE) {
        const text = node.textContent;
        node.textContent = "";
        node.appendChild(wrap(text, node.classList.contains("hl") ? "hl" : ""));
      }
    }
    const light = () => {
      const r = statement.getBoundingClientRect();
      const vh = window.innerHeight;
      const p = clamp((vh * 0.82 - r.top) / (r.height + vh * 0.3), 0, 1);
      const lit = Math.round(p * words.length);
      words.forEach((w, i) => w.classList.toggle("is-lit", i < lit));
    };
    if (reduceMotion) words.forEach((w) => w.classList.add("is-lit"));
    else {
      scrollHandlers.push(light);
      light();
    }
  }

  // Dark panels scale up to full size as they enter.
  const growers = $$("[data-grow]");
  if (!reduceMotion && growers.length) {
    const grow = () => {
      const vh = window.innerHeight;
      for (const g of growers) {
        const r = g.getBoundingClientRect();
        if (r.top > vh * 1.2 || r.bottom < -vh * 0.2) continue;
        const p = clamp((vh - r.top) / (vh * 0.7), 0, 1);
        g.style.setProperty("--grow", easeOut(p).toFixed(3));
      }
    };
    scrollHandlers.push(grow);
    grow();
  }

  /* ---------------------------------------------------- pointer niceties */

  if (finePointer && !reduceMotion) {
    for (const el of $$("[data-magnetic]")) {
      el.addEventListener("pointermove", (e) => {
        const r = el.getBoundingClientRect();
        const x = e.clientX - r.left - r.width / 2;
        const y = e.clientY - r.top - r.height / 2;
        el.style.transform = `translate(${x * 0.16}px, ${y * 0.26}px)`;
      });
      el.addEventListener("pointerleave", () => (el.style.transform = ""));
    }
    for (const el of $$("[data-tilt]")) {
      el.addEventListener("pointermove", (e) => {
        const r = el.getBoundingClientRect();
        const x = (e.clientX - r.left) / r.width - 0.5;
        const y = (e.clientY - r.top) / r.height - 0.5;
        el.style.transform = `perspective(900px) rotateX(${(-y * 7).toFixed(2)}deg) rotateY(${(x * 9).toFixed(2)}deg)`;
        el.style.boxShadow = `${-x * 24}px ${24 - y * 12}px 60px -36px rgba(11, 13, 18, 0.5)`;
      });
      el.addEventListener("pointerleave", () => {
        el.style.transform = "";
        el.style.boxShadow = "";
      });
    }
  }
  for (const card of $$("[data-spotlight]")) {
    card.addEventListener("pointermove", (e) => {
      const r = card.getBoundingClientRect();
      card.style.setProperty("--mx", `${e.clientX - r.left}px`);
      card.style.setProperty("--my", `${e.clientY - r.top}px`);
    });
  }

  // Cursor badge over media ("Zoom", "Play", "Drag").
  const badge = $("[data-cursor-badge]");
  let badgeTarget = null;
  const refreshBadge = () => {
    if (badgeTarget) badge.textContent = badgeTarget.dataset.cursor;
  };
  if (finePointer && !reduceMotion && badge) {
    let tx = -200, ty = -200, cx = tx, cy = ty, raf = 0;
    const follow = () => {
      cx = lerp(cx, tx, 0.24);
      cy = lerp(cy, ty, 0.24);
      badge.style.translate = `${cx}px ${cy}px`;
      raf = Math.abs(cx - tx) + Math.abs(cy - ty) > 0.3 ? requestAnimationFrame(follow) : 0;
    };
    document.addEventListener(
      "pointermove",
      (e) => {
        tx = e.clientX;
        ty = e.clientY;
        const t = e.target instanceof Element ? e.target.closest("[data-cursor]") : null;
        const inControls = e.target instanceof Element && e.target.closest(".video-controls");
        const next = inControls ? null : t;
        if (next !== badgeTarget) {
          if (!badgeTarget) {
            cx = tx;
            cy = ty;
          }
          badgeTarget = next;
          if (next) {
            refreshBadge();
            badge.classList.toggle("on-dark", Boolean(next.closest(".dark, .inputs")));
          }
          badge.classList.toggle("is-on", Boolean(next));
          document.body.classList.toggle("has-badge", Boolean(next));
        }
        if (!raf) raf = requestAnimationFrame(follow);
      },
      { passive: true }
    );
    document.addEventListener("pointerleave", () => {
      badgeTarget = null;
      badge.classList.remove("is-on");
      document.body.classList.remove("has-badge");
    });
  }

  /* ---------------------------------------------------- lightbox, video */

  const lightbox = $("[data-lightbox]");
  const lightboxImg = lightbox.querySelector("img");
  for (const btn of $$("[data-zoom]")) {
    btn.addEventListener("click", () => {
      if (typeof lightbox.showModal !== "function") {
        window.open(btn.dataset.zoom, "_blank", "noopener");
        return;
      }
      lightboxImg.src = btn.dataset.zoom;
      lightboxImg.alt = btn.querySelector("img")?.alt || "";
      lightbox.showModal();
    });
  }
  lightbox.addEventListener("click", () => lightbox.close());

  const videoFrame = $("[data-video]");
  if (videoFrame) {
    const video = videoFrame.querySelector("video");
    const toggle = $("[data-video-toggle]", videoFrame);
    const label = $("[data-video-label]", videoFrame);
    const full = $("[data-video-full]", videoFrame);
    let loaded = false;
    let userPaused = false;
    const load = () => {
      if (loaded) return;
      video.src = video.dataset.src;
      loaded = true;
    };
    const play = () => {
      load();
      video.play().catch(() => {});
    };
    const sync = () => {
      const playing = !video.paused;
      label.textContent = playing ? "Pause" : "Play";
      toggle.setAttribute("aria-label", playing ? "Pause video" : "Play video");
      $("[data-icon-play]", toggle).toggleAttribute("hidden", playing);
      $("[data-icon-pause]", toggle).toggleAttribute("hidden", !playing);
      videoFrame.dataset.cursor = playing ? "Pause" : "Play";
      if (badgeTarget === videoFrame) refreshBadge();
    };
    const flip = () => {
      if (video.paused) {
        userPaused = false;
        play();
      } else {
        userPaused = true;
        video.pause();
      }
    };
    video.addEventListener("play", sync);
    video.addEventListener("pause", sync);
    toggle.addEventListener("click", (e) => {
      e.stopPropagation();
      flip();
    });
    videoFrame.addEventListener("click", (e) => {
      if (!e.target.closest(".video-controls")) flip();
    });
    full.addEventListener("click", (e) => {
      e.stopPropagation();
      play();
      const enter = video.requestFullscreen || video.webkitRequestFullscreen || video.webkitEnterFullscreen;
      if (enter) enter.call(video);
    });
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(
        ([entry]) => {
          if (entry.intersectionRatio >= 0.4) {
            if (!userPaused && !reduceMotion && !saveData) play();
          } else if (!video.paused) {
            video.pause();
          }
        },
        { threshold: [0, 0.4, 0.8] }
      ).observe(videoFrame);
    }
    sync();
  }

  // BibTeX: light syntax colouring and copy.
  const bib = $("[data-bibtex]");
  if (bib) {
    const esc = bib.textContent.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);
    bib.innerHTML = esc
      .replace(/^@(\w+)\{/m, '@<span class="k">$1</span>{')
      .replace(/^(\s+)(\w+)(\s+=)/gm, '$1<span class="k">$2</span>$3')
      .replace(/(= )(\{.*\})/g, '$1<span class="s">$2</span>');
    const copyBtn = $("[data-copy]");
    const copyLabel = $("[data-copy-label]");
    copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(bib.textContent);
        copyLabel.textContent = "Copied";
      } catch {
        const range = document.createRange();
        range.selectNodeContents(bib);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        copyLabel.textContent = "Selected";
      }
      setTimeout(() => (copyLabel.textContent = "Copy BibTeX"), 1800);
    });
  }

  /* ---------------------------------------------------- point-cloud engine */

  class CloudLayer {
    constructor(n = 1024) {
      this.n = n;
      this.pos = new Float32Array(n * 3);
      this.from = new Float32Array(n * 3);
      this.to = new Float32Array(n * 3);
      this.col = new Float32Array(n * 3);
      this.colFrom = new Float32Array(n * 3);
      this.colTo = new Float32Array(n * 3);
      this.off = new Float32Array(n * 2);
      this.lift = new Float32Array(n * 3);
      this.delay = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const u = Math.random() * 2 - 1;
        const a = Math.random() * Math.PI * 2;
        const s = Math.sqrt(1 - u * u);
        this.lift[i * 3] = s * Math.cos(a);
        this.lift[i * 3 + 1] = u;
        this.lift[i * 3 + 2] = s * Math.sin(a);
      }
      this.cloud = null;
      this.pending = null;
      this.start = -1;
      this.duration = 1500;
      this.graphAlpha = 0;
      this.ready = false;
    }

    set(cloud, now, instant = false) {
      if (!this.ready || instant) {
        this.pos.set(cloud.pts);
        this.col.set(cloud.col);
        this.cloud = cloud;
        this.start = -1;
        this.ready = true;
        this.graphAlpha = instant && this.graphAlpha ? 1 : this.graphAlpha;
        return;
      }
      this.from.set(this.pos);
      this.colFrom.set(this.col);
      this.to.set(cloud.pts);
      this.colTo.set(cloud.col);
      for (let i = 0; i < this.n; i++) this.delay[i] = (i / this.n) * 0.28 + Math.random() * 0.12;
      this.pending = cloud;
      this.start = now;
    }

    get morphing() {
      return this.start >= 0;
    }

    update(now) {
      if (this.start < 0) {
        this.graphAlpha = Math.min(1, this.graphAlpha + 0.04);
        return;
      }
      this.graphAlpha = Math.max(0, this.graphAlpha - 0.12);
      const T = (now - this.start) / this.duration;
      const { pos, from, to, col, colFrom, colTo, lift, delay } = this;
      for (let i = 0; i < this.n; i++) {
        const e = easeInOut(clamp((T - delay[i]) / 0.6, 0, 1));
        const bulge = Math.sin(Math.PI * e) * 0.32;
        for (let k = 0; k < 3; k++) {
          const j = i * 3 + k;
          pos[j] = lerp(from[j], to[j], e) + lift[j] * bulge;
          col[j] = lerp(colFrom[j], colTo[j], e);
        }
      }
      if (T >= 1) {
        this.start = -1;
        this.cloud = this.pending;
        this.pending = null;
      }
    }
  }

  class Scene {
    constructor(canvas, { dprCap = 2, layout, dark = false, repel = 70, onFrame = null, autoSpin = 0.00018 }) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.dprCap = dprCap;
      this.layout = layout;
      this.dark = dark;
      this.repel = repel;
      this.onFrame = onFrame;
      this.autoSpin = reduceMotion ? 0 : autoSpin;
      this.slots = [];
      this.yaw = 0.7;
      this.pitch = -0.22;
      this.yawTarget = this.yaw;
      this.pitchTarget = this.pitch;
      this.spin = 0;
      this.pointer = { x: 0, y: 0, active: false };
      this.visible = false;
      this.raf = 0;
      this.last = 0;
      this.resize();
      if ("ResizeObserver" in window) new ResizeObserver(() => this.resize()).observe(canvas);
      if ("IntersectionObserver" in window) {
        new IntersectionObserver(([e]) => {
          this.visible = e.isIntersecting;
          if (this.visible) this.kick();
        }).observe(canvas);
      } else {
        this.visible = true;
      }
      document.addEventListener("visibilitychange", () => this.kick());
    }

    resize() {
      const r = this.canvas.getBoundingClientRect();
      if (!r.width || !r.height) return;
      this.dpr = Math.min(window.devicePixelRatio || 1, this.dprCap);
      this.canvas.width = Math.round(r.width * this.dpr);
      this.canvas.height = Math.round(r.height * this.dpr);
      this.w = r.width;
      this.h = r.height;
      this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      if (this.layout) this.layout(this);
      this.kick(true);
    }

    kick(force = false) {
      if (this.raf) return;
      if (force && (reduceMotion || !this.visible)) {
        this.frame(performance.now(), true);
        return;
      }
      if (this.visible && !document.hidden) this.raf = requestAnimationFrame((t) => this.frame(t));
    }

    frame(now, once = false) {
      this.raf = 0;
      const dt = this.last ? Math.min(now - this.last, 50) : 16;
      this.last = now;
      this.spin += dt * this.autoSpin;
      this.yaw = lerp(this.yaw, this.yawTarget + this.spin, 0.06);
      this.pitch = lerp(this.pitch, this.pitchTarget, 0.06);
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.w, this.h);
      if (this.onFrame) this.onFrame(this, now, dt);
      let busy = false;
      for (const slot of this.slots) {
        slot.layer.update(now);
        if (slot.layer.ready) this.draw(slot);
        busy = busy || slot.layer.morphing || slot.layer.graphAlpha < 1;
      }
      if (once) return;
      const keepGoing = !reduceMotion || busy || this.pointer.active;
      if (keepGoing && this.visible && !document.hidden) this.raf = requestAnimationFrame((t) => this.frame(t));
    }

    draw(slot) {
      const { layer, cx, cy, r } = slot;
      const ctx = this.ctx;
      const yaw = this.yaw + (slot.yawOffset || 0);
      const cyw = Math.cos(yaw), syw = Math.sin(yaw);
      const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
      const persp = 3.4;
      const P = layer.pos, C = layer.col, off = layer.off;
      const { x: px, y: py, active } = this.pointer;
      const R = this.repel, R2 = R * R;
      const base = slot.pointSize || 2;
      for (let i = 0; i < layer.n; i++) {
        const x = P[i * 3], y = P[i * 3 + 1], z = P[i * 3 + 2];
        const x1 = x * cyw + z * syw;
        const z1 = -x * syw + z * cyw;
        const y2 = y * cp - z1 * sp;
        const z2 = y * sp + z1 * cp;
        const k = persp / (persp - z2);
        let sx = cx + x1 * r * k;
        let sy = cy - y2 * r * k;
        let tx = 0, ty = 0;
        if (active) {
          const dx = sx - px, dy = sy - py;
          const d2 = dx * dx + dy * dy;
          if (d2 < R2) {
            const d = Math.sqrt(d2) || 1;
            const f = (1 - d / R) ** 2 * R * 0.55;
            tx = (dx / d) * f;
            ty = (dy / d) * f;
          }
        }
        off[i * 2] += (tx - off[i * 2]) * 0.14;
        off[i * 2 + 1] += (ty - off[i * 2 + 1]) * 0.14;
        sx += off[i * 2];
        sy += off[i * 2 + 1];
        const depth = (z2 + 1) / 2;
        const size = base * (0.55 + depth * 0.9);
        ctx.globalAlpha = 0.28 + depth * 0.72;
        ctx.fillStyle = `rgb(${C[i * 3] | 0},${C[i * 3 + 1] | 0},${C[i * 3 + 2] | 0})`;
        ctx.fillRect(sx - size / 2, sy - size / 2, size, size);
      }
      ctx.globalAlpha = 1;

      const g = layer.cloud;
      if (!g || !g.nodes || layer.graphAlpha <= 0.01) return;
      const proj = new Float32Array(g.nodes * 3);
      for (let k = 0; k < g.nodes; k++) {
        const x = g.cen[k * 3], y = g.cen[k * 3 + 1], z = g.cen[k * 3 + 2];
        const x1 = x * cyw + z * syw;
        const z1 = -x * syw + z * cyw;
        const y2 = y * cp - z1 * sp;
        const z2 = y * sp + z1 * cp;
        const kk = persp / (persp - z2);
        proj[k * 3] = cx + x1 * r * kk;
        proj[k * 3 + 1] = cy - y2 * r * kk;
        proj[k * 3 + 2] = z2;
      }
      const a = layer.graphAlpha;
      ctx.lineWidth = 1.25;
      ctx.strokeStyle = this.dark ? `rgba(255,255,255,${0.62 * a})` : `rgba(11,13,18,${0.55 * a})`;
      ctx.beginPath();
      for (const [i, j] of g.edges) {
        if (!g.known[i] || !g.known[j]) continue;
        ctx.moveTo(proj[i * 3], proj[i * 3 + 1]);
        ctx.lineTo(proj[j * 3], proj[j * 3 + 1]);
      }
      ctx.stroke();
      const nodeR = slot.nodeSize || 3.6;
      for (let k = 0; k < g.nodes; k++) {
        if (!g.known[k]) continue;
        const c = g.nodeCol[k];
        ctx.globalAlpha = a;
        ctx.beginPath();
        ctx.arc(proj[k * 3], proj[k * 3 + 1], nodeR, 0, Math.PI * 2);
        ctx.fillStyle = this.dark ? "#0b0d12" : "#ffffff";
        ctx.fill();
        ctx.lineWidth = 1.6;
        ctx.strokeStyle = `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
  }

  // Orbiting dashes behind the hero cloud: blue on the input side, coral on the output side.
  class DashField {
    constructor(count) {
      this.items = [];
      for (let i = 0; i < count; i++) {
        const ring = Math.random();
        this.items.push({
          a: Math.random() * Math.PI * 2,
          rr: 0.62 + ring * ring * 2.3 + Math.random() * 0.12,
          spd: (0.00005 + Math.random() * 0.00007) * (Math.random() < 0.12 ? -1 : 1),
          len: 5 + Math.random() * 7,
          ox: 0,
          oy: 0,
        });
      }
    }

    draw(scene, now, cx, cy, R, avoid) {
      const ctx = scene.ctx;
      const { x: px, y: py, active } = scene.pointer;
      const t = reduceMotion ? 0 : now;
      const buckets = new Map();
      for (const it of this.items) {
        const a = it.a + t * it.spd * (1.4 / Math.sqrt(it.rr));
        const breathe = 1 + Math.sin(t * 0.0006 + it.rr * 4) * 0.02;
        const rx = it.rr * R * 1.5 * breathe;
        const ry = it.rr * R * 0.95 * breathe;
        let x = cx + Math.cos(a) * rx;
        let y = cy + Math.sin(a) * ry;
        let tx = 0, ty = 0;
        if (active) {
          const dx = x - px, dy = y - py;
          const d = Math.hypot(dx, dy);
          if (d < 140) {
            const f = (1 - d / 140) ** 2 * 60;
            tx = (dx / (d || 1)) * f;
            ty = (dy / (d || 1)) * f;
          }
        }
        it.ox += (tx - it.ox) * 0.08;
        it.oy += (ty - it.oy) * 0.08;
        x += it.ox;
        y += it.oy;
        if (x < -20 || y < -20 || x > scene.w + 20 || y > scene.h + 20) continue;
        let dx = -Math.sin(a) * rx, dy = Math.cos(a) * ry;
        const dl = Math.hypot(dx, dy) || 1;
        dx = (dx / dl) * it.len * 0.5;
        dy = (dy / dl) * it.len * 0.5;
        const hue = Math.round(clamp((x - (cx - rx)) / (2 * rx || 1), 0, 1) * 10);
        let alpha = it.rr < 1.1 ? 2 : it.rr < 1.8 ? 1 : 0;
        if (avoid && x > avoid.left && x < avoid.right && y > avoid.top && y < avoid.bottom) alpha = 0;
        const key = hue * 3 + alpha;
        let list = buckets.get(key);
        if (!list) buckets.set(key, (list = []));
        list.push(x - dx, y - dy, x + dx, y + dy);
      }
      ctx.lineWidth = 1.4;
      ctx.lineCap = "round";
      for (const [key, list] of buckets) {
        const hue = Math.floor(key / 3);
        const alpha = [0.16, 0.34, 0.5][key % 3];
        const c = gradient(hue / 10);
        ctx.strokeStyle = `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${alpha})`;
        ctx.beginPath();
        for (let i = 0; i < list.length; i += 4) {
          ctx.moveTo(list[i], list[i + 1]);
          ctx.lineTo(list[i + 2], list[i + 3]);
        }
        ctx.stroke();
      }
    }
  }

  /* ---------------------------------------------------------------- hero */

  const heroCanvas = $("[data-hero-canvas]");
  const heroStage = $("[data-hero-stage]");
  const hero = $(".hero");
  const PHRASES = [
    { text: "dexterous hands", hand: "Allegro" },
    { text: "zero-shot end effectors", hand: "FreedomHand", note: "zero-shot in the paper" },
    { text: "parallel grippers", hand: "wsg_50" },
    { text: "three-finger hands", hand: "Barrett" },
  ];

  if (heroCanvas && heroStage) {
    const layer = new CloudLayer();
    const field = new DashField(window.innerWidth < 720 ? 120 : 260);
    const copy = $(".hero-copy");
    let avoid = null;
    const heroScene = new Scene(heroCanvas, {
      dprCap: 1.5,
      repel: 80,
      autoSpin: 0.00022,
      layout(scene) {
        const cr = heroCanvas.getBoundingClientRect();
        const sr = heroStage.getBoundingClientRect();
        const r = Math.min(sr.width * 0.36, sr.height * 0.4, 280);
        const slot = { layer, cx: sr.left - cr.left + sr.width / 2, cy: sr.top - cr.top + sr.height * 0.47, r, pointSize: r > 200 ? 2.4 : 2, nodeSize: r > 200 ? 4 : 3.2 };
        scene.slots = [slot];
        scene.fieldR = r;
        const pr = copy.getBoundingClientRect();
        avoid = window.innerWidth > 1080 ? { left: pr.left - cr.left - 12, right: pr.right - cr.left + 12, top: pr.top - cr.top - 12, bottom: pr.bottom - cr.top + 12 } : null;
      },
      onFrame(scene, now) {
        const slot = scene.slots[0];
        if (slot) field.draw(scene, now, slot.cx, slot.cy, scene.fieldR, avoid);
      },
    });

    if (!reduceMotion) {
      hero.addEventListener("pointermove", (e) => {
        const cr = heroCanvas.getBoundingClientRect();
        const slot = heroScene.slots[0];
        heroScene.pointer.x = e.clientX - cr.left;
        heroScene.pointer.y = e.clientY - cr.top;
        heroScene.pointer.active = e.pointerType === "mouse";
        if (slot && e.pointerType === "mouse") {
          const nx = clamp((heroScene.pointer.x - slot.cx) / (cr.width * 0.5), -1, 1);
          const ny = clamp((heroScene.pointer.y - slot.cy) / (cr.height * 0.5), -1, 1);
          heroScene.yawTarget = 0.7 + nx * 1.1;
          heroScene.pitchTarget = -0.22 + ny * 0.5;
        }
        heroScene.kick();
      });
      hero.addEventListener("pointerleave", () => {
        heroScene.pointer.active = false;
        heroScene.yawTarget = 0.7;
        heroScene.pitchTarget = -0.22;
      });
    }

    const nameEl = $("[data-hero-name]");
    const metaEl = $("[data-hero-meta]");
    const hintEl = $("[data-hero-hint]");
    const typedEl = $("[data-typed]");
    if (!finePointer) hintEl.textContent = "Tap to switch";
    let index = 0;
    let data = null;
    let timer = 0;

    const describe = (phrase) => {
      const c = data.hands[phrase.hand];
      nameEl.textContent = nameOf(HANDS, phrase.hand);
      const meta = `1,024-point cloud · ${c.nodes}-node graph`;
      metaEl.textContent = phrase.note ? `${meta} · ${phrase.note}` : meta;
    };

    const typeTo = (text, done) => {
      if (reduceMotion) {
        typedEl.textContent = text;
        done();
        return;
      }
      const current = typedEl.textContent;
      let i = current.length;
      const erase = () => {
        if (i > 0) {
          typedEl.textContent = current.slice(0, --i);
          timer = setTimeout(erase, 22);
          return;
        }
        let j = 0;
        const type = () => {
          typedEl.textContent = text.slice(0, ++j);
          if (j < text.length) timer = setTimeout(type, 42 + Math.random() * 30);
          else done();
        };
        type();
      };
      erase();
    };

    const advance = () => {
      clearTimeout(timer);
      index = (index + 1) % PHRASES.length;
      const phrase = PHRASES[index];
      layer.set(data.hands[phrase.hand], performance.now(), reduceMotion);
      describe(phrase);
      heroScene.kick();
      typeTo(phrase.text, () => {
        timer = setTimeout(schedule, 3600);
      });
    };
    const schedule = () => {
      // Hold the cycle while the hero is off screen.
      if (!heroScene.visible || document.hidden) {
        timer = setTimeout(schedule, 800);
        return;
      }
      advance();
    };

    cloudsReady.then((d) => {
      if (!d) return;
      data = d;
      layer.set(d.hands[PHRASES[0].hand], performance.now(), true);
      describe(PHRASES[0]);
      heroScene.layout(heroScene);
      heroScene.kick(true);
      heroScene.kick();
      timer = setTimeout(schedule, reduceMotion ? 6000 : 3800);
      heroStage.addEventListener("click", advance);
    });
  }

  /* ------------------------------------------------------------ marquee */

  const tileFor = (hand, object) => {
    const a = document.createElement("a");
    a.className = "m-tile";
    a.href = "#explorer";
    a.tabIndex = -1;
    a.dataset.hand = hand;
    a.dataset.object = object;
    a.innerHTML = `<img alt="" width="56" height="56" loading="lazy"><span><b></b><span></span></span>`;
    a.querySelector("img").src = render(hand, object, 1);
    a.querySelector("b").textContent = nameOf(HANDS, hand);
    a.querySelector("span > span").textContent = nameOf(OBJECTS, object);
    return a;
  };
  const marqueeRows = {
    hands: HANDS.map(([h], i) => [h, OBJECTS[(i * 2 + 6) % OBJECTS.length][0]]),
    objects: HANDS.map((_, i) => [HANDS[(i * 4 + 3) % HANDS.length][0], OBJECTS[(i * 3 + 1) % OBJECTS.length][0]]),
  };
  for (const el of $$("[data-marquee]")) {
    const track = document.createElement("div");
    track.className = "marquee-track";
    for (const [h, o] of marqueeRows[el.dataset.marquee]) track.appendChild(tileFor(h, o));
    el.appendChild(track);
    el.appendChild(track.cloneNode(true));
  }

  /* ------------------------------------------------------ method walkthrough */

  const steps = $$("[data-steps] .step");
  const viewport = $("[data-figure-viewport]");
  if (steps.length && viewport) {
    const img = viewport.querySelector("img");
    const regions = steps.map((s) => s.dataset.region.split(",").map(Number));
    const label = $("[data-step-label]");
    const bar = $(".method-meta .bar span");
    const dots = $("[data-step-dots]");
    const last = steps.length - 1;
    let active = -1;

    steps.forEach((step, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = String(i).padStart(2, "0");
      b.setAttribute("aria-label", `Go to step ${i}: ${step.querySelector("h3").textContent}`);
      b.addEventListener("click", () => step.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "center" }));
      dots.appendChild(b);
    });

    const frame = () => {
      const [x0, y0, x1, y1] = regions[Math.max(active, 0)];
      const vw = viewport.clientWidth, vh = viewport.clientHeight;
      const pad = active <= 0 ? 1.04 : 1.1;
      const s = Math.min(vw / ((x1 - x0) * pad), vh / ((y1 - y0) * pad));
      const tx = vw / 2 - ((x0 + x1) / 2) * s;
      const ty = vh / 2 - ((y0 + y1) / 2) * s;
      img.style.transform = `translate(${tx.toFixed(1)}px, ${ty.toFixed(1)}px) scale(${s.toFixed(4)})`;
    };

    const setActive = (i) => {
      if (i === active) return;
      active = i;
      steps.forEach((s, k) => s.classList.toggle("is-active", k === i));
      [...dots.children].forEach((d, k) => (k === i ? d.setAttribute("aria-current", "true") : d.removeAttribute("aria-current")));
      label.textContent = `${String(i).padStart(2, "0")} / ${String(last).padStart(2, "0")}`;
      bar.parentElement.style.setProperty("--p", "");
      bar.style.setProperty("--p", Math.max(0.06, i / last).toFixed(3));
      frame();
    };

    const pick = () => {
      const narrow = window.innerWidth <= 1080;
      const line = window.innerHeight * (narrow ? 0.74 : 0.5);
      let best = 0;
      steps.forEach((s, i) => {
        if (s.getBoundingClientRect().top <= line) best = i;
      });
      setActive(best);
    };
    scrollHandlers.push(pick);
    window.addEventListener("resize", frame);
    if ("ResizeObserver" in window) new ResizeObserver(frame).observe(viewport);
    pick();
  }

  /* ---------------------------------------------------------- explorer */

  const explorer = $("[data-explorer]");
  if (explorer) {
    const state = { hand: "Allegro", object: "025_mug" };
    const handStrip = $("[data-hand-strip]", explorer);
    const objectStrip = $("[data-object-strip]", explorer);
    const shots = $$(".shot img", explorer);
    const graphMeta = $("[data-graph-meta]", explorer);
    const inputs = $("[data-inputs]", explorer);
    const inputsCanvas = $("[data-inputs-canvas]", explorer);
    let data = null;

    const objLayer = new CloudLayer();
    const handLayer = new CloudLayer();
    const scene = new Scene(inputsCanvas, {
      dark: true,
      repel: 50,
      autoSpin: 0.00016,
      layout(s) {
        const r = Math.min(s.w * 0.225, s.h * 0.34);
        const size = r > 90 ? 2.2 : 1.8;
        s.slots = [
          { layer: objLayer, cx: s.w * 0.27, cy: s.h * 0.5, r: r * 0.9, pointSize: size },
          { layer: handLayer, cx: s.w * 0.73, cy: s.h * 0.5, r, pointSize: size, nodeSize: r > 90 ? 3.4 : 2.8 },
        ];
      },
      onFrame(s) {
        // A faint "+" between the two inputs.
        const ctx = s.ctx;
        const x = s.w * 0.5, y = s.h * 0.5;
        ctx.strokeStyle = "rgba(255,255,255,0.35)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x - 7, y);
        ctx.lineTo(x + 7, y);
        ctx.moveTo(x, y - 7);
        ctx.lineTo(x, y + 7);
        ctx.stroke();
      },
    });

    // Drag to rotate; hover pushes points aside.
    let drag = null;
    inputs.addEventListener("pointerdown", (e) => {
      drag = { x: e.clientX, y: e.clientY, yaw: scene.yawTarget, pitch: scene.pitchTarget };
      inputs.setPointerCapture(e.pointerId);
    });
    inputs.addEventListener("pointermove", (e) => {
      const r = inputsCanvas.getBoundingClientRect();
      scene.pointer.x = e.clientX - r.left;
      scene.pointer.y = e.clientY - r.top;
      scene.pointer.active = e.pointerType === "mouse" && !reduceMotion;
      if (drag) {
        scene.yawTarget = drag.yaw + (e.clientX - drag.x) * 0.01;
        scene.pitchTarget = clamp(drag.pitch + (e.clientY - drag.y) * 0.006, -1.1, 0.7);
      }
      scene.kick();
    });
    const endDrag = () => (drag = null);
    inputs.addEventListener("pointerup", endDrag);
    inputs.addEventListener("pointercancel", endDrag);
    inputs.addEventListener("pointerleave", () => (scene.pointer.active = false));

    const setImage = (img, src, alt) => {
      if (img.getAttribute("src") === src) return;
      img.classList.add("is-loading");
      const pre = new Image();
      pre.onload = pre.onerror = () => {
        img.src = src;
        img.alt = alt;
        img.classList.remove("is-loading");
      };
      pre.src = src;
    };

    const buildStrip = (strip, list, kind) => {
      for (const [key, name] of list) {
        const li = document.createElement("li");
        li.innerHTML = `<button class="tile" type="button" aria-pressed="false"><span class="tile-img"><img width="360" height="360" alt="" loading="lazy"></span><span class="tile-name"></span></button>`;
        const btn = li.querySelector("button");
        btn.dataset.key = key;
        btn.title = name;
        li.querySelector(".tile-name").textContent = name;
        btn.addEventListener("click", () => {
          state[kind] = key;
          update();
        });
        strip.appendChild(li);
      }
    };

    const keepInView = (strip, smooth) => {
      const on = strip.querySelector('[aria-pressed="true"]');
      if (!on || strip.scrollWidth <= strip.clientWidth) return;
      const li = on.parentElement;
      strip.scrollTo({ left: li.offsetLeft - (strip.clientWidth - li.offsetWidth) / 2, behavior: smooth ? "smooth" : "auto" });
    };

    function update(smooth = true) {
      const hand = nameOf(HANDS, state.hand);
      const object = nameOf(OBJECTS, state.object);
      const lower = object.toLowerCase();
      $$("[data-hand-name]", explorer).forEach((el) => (el.textContent = hand));
      $$("[data-object-name]", explorer).forEach((el) => (el.textContent = object));
      $$("[data-object-lower]", explorer).forEach((el) => (el.textContent = lower));
      shots.forEach((img, i) => setImage(img, render(state.hand, state.object, i + 1), `${hand} grasping the ${lower}, rank ${i + 1}`));
      for (const tile of $$(".tile", handStrip)) {
        const k = tile.dataset.key;
        tile.setAttribute("aria-pressed", String(k === state.hand));
        tile.setAttribute("aria-label", `${nameOf(HANDS, k)} grasping the ${lower}`);
        setImage(tile.querySelector("img"), render(k, state.object, 1), "");
      }
      for (const tile of $$(".tile", objectStrip)) {
        const k = tile.dataset.key;
        tile.setAttribute("aria-pressed", String(k === state.object));
        tile.setAttribute("aria-label", `${hand} grasping the ${nameOf(OBJECTS, k).toLowerCase()}`);
        setImage(tile.querySelector("img"), render(state.hand, k, 1), "");
      }
      keepInView(handStrip, smooth);
      keepInView(objectStrip, smooth);
      if (data) {
        const h = data.hands[state.hand];
        const now = performance.now();
        handLayer.set(h, now, reduceMotion);
        objLayer.set(data.objects[state.object], now, reduceMotion);
        graphMeta.textContent = `${h.nodes} nodes · ${h.edges.length} edges`;
        scene.kick();
      }
    }

    buildStrip(handStrip, HANDS, "hand");
    buildStrip(objectStrip, OBJECTS, "object");
    update(false);

    cloudsReady.then((d) => {
      if (!d) return;
      data = d;
      update(false);
      scene.kick(true);
    });

    // Marquee tiles jump to their pair in the explorer.
    for (const tile of $$(".m-tile")) {
      tile.addEventListener("click", (e) => {
        e.preventDefault();
        state.hand = tile.dataset.hand;
        state.object = tile.dataset.object;
        update(false);
        $("#explorer").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
      });
    }
  }

  /* ------------------------------------------------ footer particle wordmark */

  const footerCanvas = $("[data-footer-canvas]");
  if (footerCanvas) {
    const ctx = footerCanvas.getContext("2d");
    let parts = [];
    let buckets = [];
    let w = 0, h = 0, dpr = 1, step = 5;
    let visible = false, raf = 0, assembled = false;
    const pointer = { x: -999, y: -999, active: false };

    const build = () => {
      const r = footerCanvas.getBoundingClientRect();
      if (!r.width) return;
      w = r.width;
      h = r.height;
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      footerCanvas.width = Math.round(w * dpr);
      footerCanvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const off = document.createElement("canvas");
      off.width = Math.round(w);
      off.height = Math.round(h);
      const o = off.getContext("2d");
      let size = h * 1.18;
      o.font = `700 ${size}px Geist, ui-sans-serif, system-ui, sans-serif`;
      const measured = o.measureText("EAGG").width;
      if (measured > w * 0.98) size *= (w * 0.98) / measured;
      o.font = `700 ${size}px Geist, ui-sans-serif, system-ui, sans-serif`;
      o.textAlign = "center";
      o.textBaseline = "alphabetic";
      o.fillStyle = "#000";
      o.fillText("EAGG", w / 2, h - size * 0.04);
      const img = o.getImageData(0, 0, off.width, off.height).data;
      step = clamp(Math.round(w / 190), 3, 7);
      const prev = parts;
      parts = [];
      for (let y = 0; y < off.height; y += step) {
        for (let x = 0; x < off.width; x += step) {
          if (img[(y * off.width + x) * 4 + 3] < 128) continue;
          const i = parts.length;
          const p = prev[i];
          parts.push({
            tx: x,
            ty: y,
            x: p ? p.x : Math.random() * w,
            y: p ? p.y : h + Math.random() * h * 0.8,
            vx: 0,
            vy: 0,
          });
        }
      }
      // Particles are laid out left to right, so colour buckets are contiguous runs.
      parts.sort((a, b) => a.tx - b.tx);
      buckets = [];
      const n = 24;
      for (let b = 0; b < n; b++) buckets.push({ color: gradient(b / (n - 1)), from: 0, to: 0 });
      let bi = 0;
      parts.forEach((p, i) => {
        const b = Math.min(n - 1, Math.floor((p.tx / w) * n));
        while (bi < b) {
          buckets[bi].to = i;
          bi++;
          buckets[bi].from = i;
        }
      });
      for (; bi < n; bi++) {
        buckets[bi].to = parts.length;
        if (bi + 1 < n) buckets[bi + 1].from = parts.length;
      }
      if (reduceMotion) {
        parts.forEach((p) => {
          p.x = p.tx;
          p.y = p.ty;
        });
        assembled = true;
      }
      draw();
    };

    const draw = () => {
      ctx.clearRect(0, 0, w, h);
      const s = step * 0.62;
      for (const b of buckets) {
        ctx.fillStyle = `rgb(${b.color[0] | 0},${b.color[1] | 0},${b.color[2] | 0})`;
        for (let i = b.from; i < b.to; i++) {
          const p = parts[i];
          ctx.fillRect(p.x - s / 2, p.y - s / 2, s, s);
        }
      }
    };

    const tick = () => {
      raf = 0;
      let energy = 0;
      const R = Math.max(70, w * 0.07);
      for (const p of parts) {
        let ax = (p.tx - p.x) * (assembled ? 0.07 : 0.035);
        let ay = (p.ty - p.y) * (assembled ? 0.07 : 0.035);
        if (pointer.active) {
          const dx = p.x - pointer.x, dy = p.y - pointer.y;
          const d2 = dx * dx + dy * dy;
          if (d2 < R * R) {
            const d = Math.sqrt(d2) || 1;
            const f = (1 - d / R) * 5;
            ax += (dx / d) * f;
            ay += (dy / d) * f;
          }
        }
        p.vx = (p.vx + ax) * 0.8;
        p.vy = (p.vy + ay) * 0.8;
        p.x += p.vx;
        p.y += p.vy;
        energy += Math.abs(p.vx) + Math.abs(p.vy);
      }
      draw();
      if (energy / (parts.length || 1) < 0.02 && !pointer.active) {
        assembled = true;
        return;
      }
      if (visible && !document.hidden) raf = requestAnimationFrame(tick);
    };
    const kick = () => {
      if (!raf && visible && !reduceMotion) raf = requestAnimationFrame(tick);
    };

    const fontReady = document.fonts && document.fonts.load ? Promise.race([document.fonts.load("700 100px Geist"), new Promise((r) => setTimeout(r, 2500))]) : Promise.resolve();
    fontReady.then(() => {
      build();
      if ("IntersectionObserver" in window) {
        new IntersectionObserver(([e]) => {
          visible = e.isIntersecting;
          if (visible) kick();
        }, { threshold: 0.2 }).observe(footerCanvas);
      }
      let lastW = w;
      if ("ResizeObserver" in window) {
        new ResizeObserver(() => {
          const nw = footerCanvas.getBoundingClientRect().width;
          if (Math.abs(nw - lastW) < 2) return;
          lastW = nw;
          build();
          kick();
        }).observe(footerCanvas);
      }
    });

    if (!reduceMotion) {
      footerCanvas.parentElement.addEventListener("pointermove", (e) => {
        const r = footerCanvas.getBoundingClientRect();
        pointer.x = e.clientX - r.left;
        pointer.y = e.clientY - r.top;
        pointer.active = true;
        kick();
      });
      footerCanvas.parentElement.addEventListener("pointerleave", () => {
        pointer.active = false;
        kick();
      });
    }
  }

  runScroll();
})();

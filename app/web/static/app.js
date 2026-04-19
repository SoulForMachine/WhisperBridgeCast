const entriesBox = document.getElementById("entries");
const jumpLink = document.getElementById("jump-bottom");
const sourceToggleLink = document.getElementById("toggle-source");
const diffToggleLink = document.getElementById("toggle-diff");
let wakeLock = null;
let showSource = true;
let showDiffHighlights = true;

function atBottom() {
  const threshold = 100;
  return window.innerHeight + window.scrollY >= document.body.scrollHeight - threshold;
}

const entriesById = new Map();
const clamp = (v, min, max) => Math.min(Math.max(v, min), max);

function isPunctuationChar(ch) {
  return Boolean(ch) && /[^\w\s]/.test(ch);
}

function pickRemovalMarkerIndex(text, idx) {
  idx = clamp(idx, 0, text.length - 1);
  if (isPunctuationChar(text[idx])) {
    return idx;
  }
  return idx > 0 && text[idx - 1] === " " ? idx - 1 : idx;
}

function refreshEntryVisibility(entry) {
  const hasOrig = Boolean(entry.querySelector(".orig-row"));
  const hasTransl = Boolean(entry.querySelector(".transl-row"));
  entry.classList.toggle("only-source", hasOrig && !hasTransl);
}

function refreshEntrySeparator(entry) {
  const hasOrig = Boolean(entry.querySelector(".orig-row"));
  const hasTransl = Boolean(entry.querySelector(".transl-row"));
  let sep = entry.querySelector(".sep");

  if (hasOrig && hasTransl) {
    if (!sep) {
      sep = document.createElement("hr");
      sep.className = "sep";
      const translRow = entry.querySelector(".transl-row");
      entry.insertBefore(sep, translRow);
    }
  } else if (sep) {
    sep.remove();
  }
}

function applySourceVisibility() {
  document.body.classList.toggle("hide-source", !showSource);
  sourceToggleLink.textContent = showSource ? "Hide" : "View";
}

function applyDiffHighlighting() {
  diffToggleLink.textContent = showDiffHighlights ? "Hide" : "View";
}

function getOrCreateEntry(entryId) {
  if (entriesById.has(entryId)) {
    return entriesById.get(entryId);
  }

  const entry = document.createElement("div");
  entry.className = "entry";
  entry.dataset.entryId = String(entryId);
  entriesBox.appendChild(entry);
  entriesById.set(entryId, entry);
  return entry;
}

function renderDiffText(body, text, diffOps, unconfirmedFrom = -1, highlightDiffs = true) {
  body.replaceChildren();

  if (!text) {
    return;
  }

  const textLen = text.length;
  const removeByIdx = new Map();
  const removeMarkerIdx = new Set();
  const addSpans = [];
  const replaceSpans = [];

  for (const op of diffOps || []) {
    if (!op || !op.op) {
      continue;
    }

    if (op.op === "-" && Number.isInteger(op.idx)) {
      const idx = clamp(op.idx, 0, textLen);
      removeByIdx.set(idx, (removeByIdx.get(idx) || 0) + 1);
      continue;
    }

    if ((op.op === "+" || op.op === "~") && Array.isArray(op.span) && op.span.length === 2) {
      const start = clamp(op.span[0], 0, textLen);
      const end = clamp(op.span[1], 0, textLen);
      if (start < end) {
        if (op.op === "+") {
          addSpans.push([start, end]);
        } else {
          replaceSpans.push([start, end]);
        }
      }
    }
  }

  const boundaries = new Set([0, textLen]);
  const split = clamp(unconfirmedFrom, 0, textLen);
  boundaries.add(split);

  for (const [idx] of removeByIdx) {
    const markerIdx = pickRemovalMarkerIndex(text, idx);
    if (markerIdx >= 0 && markerIdx < textLen) {
      removeMarkerIdx.add(markerIdx);
      boundaries.add(markerIdx);
      boundaries.add(markerIdx + 1);
    }
  }

  for (const [start, end] of addSpans) {
    boundaries.add(start);
    boundaries.add(end);
  }
  for (const [start, end] of replaceSpans) {
    boundaries.add(start);
    boundaries.add(end);
  }
  for (const idx of removeByIdx.keys()) {
    boundaries.add(idx);
  }

  const points = Array.from(boundaries).sort((a, b) => a - b);

  const hasRange = (ranges, segStart, segEnd) => ranges.some(([start, end]) => start <= segStart && segEnd <= end);

  for (let i = 0; i < points.length - 1; i += 1) {
    const segStart = points[i];
    const segEnd = points[i + 1];

    if (segStart >= segEnd) {
      continue;
    }

    const chunk = text.slice(segStart, segEnd);
    const span = document.createElement("span");
    span.textContent = chunk;

    if (highlightDiffs) {
      if (hasRange(replaceSpans, segStart, segEnd)) {
        span.classList.add("diff-replace");
      } else if (hasRange(addSpans, segStart, segEnd)) {
        span.classList.add("diff-add");
      }

      if (segEnd === segStart + 1 && removeMarkerIdx.has(segStart)) {
        span.classList.add("diff-remove");
      }
    }

    if (unconfirmedFrom >= 0 && segStart >= unconfirmedFrom) {
      span.classList.add("unconfirmed");
    }

    body.appendChild(span);
  }
}

function upsertRow(entry, kind, langCode, text, diffOps, unconfirmedFrom = -1, highlightDiffs = true) {
  const rowClass = `${kind}-row`;
  let row = entry.querySelector(`.${rowClass}`);

  if (!text) {
    if (row) {
      row.remove();
    }
    return;
  }

  if (!row) {
    row = document.createElement("div");
    row.className = `row ${rowClass}`;

    const pill = document.createElement("span");
    pill.className = `pill ${kind === "orig" ? "orig" : "transl"}`;
    row.appendChild(pill);

    const body = document.createElement("div");
    body.className = "content";
    row.appendChild(body);

    entry.appendChild(row);
  }

  const pill = row.querySelector(".pill");
  const body = row.querySelector(".content");
  pill.textContent = langCode.split('-')[0] || "";
  renderDiffText(body, text, diffOps, unconfirmedFrom, highlightDiffs);
}

function renderEntry(payload) {
  const stick = atBottom();
  const entry = getOrCreateEntry(payload.entry_id);

  const hasOriginal = Object.prototype.hasOwnProperty.call(payload, "original") ||
    Object.prototype.hasOwnProperty.call(payload, "unconfirmed");
  const hasTranslation = Object.prototype.hasOwnProperty.call(payload, "translation");

  if (hasOriginal) {
    const confirmed = payload.original || "";
    const unconfirmed = payload.unconfirmed || "";
    upsertRow(
      entry,
      "orig",
      payload.src_lang,
      `${confirmed}${unconfirmed}`,
      payload.source_diff || [],
      confirmed.length,
      showDiffHighlights,
    );
  }

  if (hasTranslation) {
    upsertRow(
      entry,
      "transl",
      payload.target_lang,
      payload.translation || "",
      payload.target_diff || [],
      -1,
      showDiffHighlights,
    );
  }

  refreshEntryVisibility(entry);
  refreshEntrySeparator(entry);

  if (stick) {
    window.scrollTo({ top: document.documentElement.scrollHeight, behavior: "smooth" });
  }
}

const es = new EventSource("/events");
es.onmessage = (event) => {
  try {
    const payload = JSON.parse(event.data);
    if (payload.type === "entry") {
      renderEntry(payload);
    }
  } catch (err) {
    console.error("Bad event", err);
  }
};

es.onerror = () => {
  console.warn("Connection lost, retrying...");
};

jumpLink.addEventListener("click", (e) => {
  e.preventDefault();
  window.scrollTo({ top: document.documentElement.scrollHeight, behavior: "smooth" });
});

sourceToggleLink.addEventListener("click", (e) => {
  e.preventDefault();
  showSource = !showSource;
  applySourceVisibility();
});

diffToggleLink.addEventListener("click", (e) => {
  e.preventDefault();
  showDiffHighlights = !showDiffHighlights;
  applyDiffHighlighting();
});

applySourceVisibility();
applyDiffHighlighting();

async function requestWakeLock() {
  try {
    if ("wakeLock" in navigator) {
      wakeLock = await navigator.wakeLock.request("screen");
      wakeLock.addEventListener("release", () => console.log("WakeLock released"));
    }
  } catch (err) {
    console.warn("WakeLock not available:", err);
  }
}

document.addEventListener("visibilitychange", async () => {
  if (document.visibilityState === "visible") {
    await requestWakeLock();
  }
});

document.addEventListener("pointerdown", async () => {
  await requestWakeLock();
});

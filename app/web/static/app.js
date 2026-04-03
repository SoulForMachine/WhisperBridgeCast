const entriesBox = document.getElementById("entries");
const jumpLink = document.getElementById("jump-bottom");
const sourceToggleLink = document.getElementById("toggle-source");
let wakeLock = null;
let showSource = true;

function atBottom() {
  const threshold = 100;
  return window.innerHeight + window.scrollY >= document.body.scrollHeight - threshold;
}

const entriesById = new Map();

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

function upsertRow(entry, kind, langCode, confirmedText, unconfirmedText) {
  const rowClass = `${kind}-row`;
  let row = entry.querySelector(`.${rowClass}`);

  if (!(confirmedText || unconfirmedText)) {
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
  pill.textContent = langCode || "";
  body.replaceChildren();

  if (confirmedText) {
    const confirmedSpan = document.createElement("span");
    confirmedSpan.textContent = confirmedText;
    body.appendChild(confirmedSpan);
  }

  if (unconfirmedText) {
    const unconfirmedSpan = document.createElement("span");
    unconfirmedSpan.className = "unconfirmed";
    unconfirmedSpan.textContent = unconfirmedText;
    body.appendChild(unconfirmedSpan);
  }
}

function renderEntry(payload) {
  const stick = atBottom();
  const entry = getOrCreateEntry(payload.entry_id);

  const hasOriginal = Object.prototype.hasOwnProperty.call(payload, "original") ||
    Object.prototype.hasOwnProperty.call(payload, "unconfirmed");
  const hasTranslation = Object.prototype.hasOwnProperty.call(payload, "translation");

  if (hasOriginal) {
    upsertRow(entry, "orig", payload.src_lang, payload.original, payload.unconfirmed);
  }

  if (hasTranslation) {
    upsertRow(entry, "transl", payload.target_lang, payload.translation, "");
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

applySourceVisibility();

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

const TAU = Math.PI * 2;
const SAMPLE_COUNT = 720;
const VISUALIZER_SIZE = 400;
const VISUALIZER_CENTER = VISUALIZER_SIZE / 2;
const PERIMETER_BASE_RADIUS = 132;
const ORBIT_PERIOD_MS = 2000;
const DISPLACEMENT_SCALE = 300;
const BASE_DECAY = 0.992;
const SILENCE_DECAY = 0.968;
const SILENCE_THRESHOLD = 0.018;
const POLL_INTERVAL_MS = 1800;
const MUSIC_BUFFER_MAX_SECONDS = 10;
const DISPLACEMENT_MOOD_SCALE = {
  default: 1,
  tense: 1.06,
  action: 1.16,
  boss: 1.1,
  mystery: 0.96,
  victory: 0.92,
};

const elements = {
  shell: document.getElementById("shell"),
  perimeterShadow: document.getElementById("perimeter-shadow"),
  perimeterFill: document.getElementById("perimeter-fill"),
  perimeterHighlight: document.getElementById("perimeter-highlight"),
  enginePill: document.getElementById("engine-pill"),
  subtitle: document.getElementById("subtitle"),
  statusBanner: document.getElementById("status-banner"),
  apiBaseInput: document.getElementById("api-base-input"),
  languageInput: document.getElementById("language-input"),
  chunkSecondsInput: document.getElementById("chunk-seconds-input"),
  moodAgentToggle: document.getElementById("mood-agent-toggle"),
  musicToggle: document.getElementById("music-toggle"),
  musicModeSelect: document.getElementById("music-mode-select"),
  musicVolumeInput: document.getElementById("music-volume-input"),
  refreshButton: document.getElementById("refresh-button"),
  sessionToggle: document.getElementById("session-toggle"),
  micButtonIcon: document.getElementById("mic-button-icon"),
  micButtonLabel: document.getElementById("mic-button-label"),
  tooltip: document.getElementById("tooltip"),
  metricMood: document.getElementById("metric-mood"),
  metricConfidence: document.getElementById("metric-confidence"),
  metricUtterances: document.getElementById("metric-utterances"),
  metricMusic: document.getElementById("metric-music"),
  tags: document.getElementById("tags"),
  moodReason: document.getElementById("mood-reason"),
  moodGrid: document.getElementById("mood-grid"),
  utteranceList: document.getElementById("utterance-list"),
  transcriptMeta: document.getElementById("transcript-meta"),
  transcriptBox: document.getElementById("transcript-box"),
  musicStatusCopy: document.getElementById("music-status-copy"),
  musicPrompt: document.getElementById("music-prompt"),
  musicReason: document.getElementById("music-reason"),
  musicPalette: document.getElementById("music-palette"),
  logBox: document.getElementById("log-box"),
};

const preview = {
  mode: "demo",
  permission: "idle",
  stream: null,
  context: null,
  source: null,
  analyser: null,
  timeData: null,
  processor: null,
  flushTimerId: 0,
  chunkBuffers: [],
  chunkSequence: 0,
  activeSessionId: "",
  uploadQueue: [],
  uploadInFlight: false,
  stopping: false,
  animationId: 0,
  displacementSamples: new Array(SAMPLE_COUNT).fill(0),
  smoothedDisplacement: 0,
};

const player = {
  context: null,
  processor: null,
  gainNode: null,
  websocket: null,
  queue: [],
  currentChunk: null,
  currentFrameOffset: 0,
  bufferedFrames: 0,
  sampleRate: 48000,
  channels: 2,
  status: "idle",
  manualClose: false,
};

const appState = {
  runtime: {
    running: false,
    current_session_id: null,
    processing: false,
    config: {
      language: "en",
      transcription_model: "voxtral-mini-latest",
      mood_model: "mistral-small-latest",
      mood_agent: true,
      mood_window: 6,
      chunk_seconds: 2.5,
      music_enabled: true,
      music_mode: "raw",
      music_temperature: 1,
      music_guidance: 4,
      music_crossfade_seconds: 4,
      music_crossfade_steps: 8,
      music_seed: 42,
      music_prompt_model: "mistral-small-latest",
    },
    music: {
      enabled: false,
      available: false,
      connected: false,
      mode: "raw",
      prompt: "",
      palette_summary: "No palette",
      last_reason: "",
      error: "",
    },
    logs: [],
  },
  session: {
    allowed_moods: [],
    utterances: [],
    mood_history: [],
    music: {},
  },
  error: "",
};

function titleCase(value) {
  return String(value || "default")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function getMoodScale() {
  return DISPLACEMENT_MOOD_SCALE[appState.session.current_mood] || DISPLACEMENT_MOOD_SCALE.default;
}

function midpoint(pointA, pointB) {
  return {
    x: (pointA.x + pointB.x) / 2,
    y: (pointA.y + pointB.y) / 2,
  };
}

function buildClosedPath(points) {
  if (points.length < 3) {
    return "";
  }

  const start = midpoint(points[points.length - 1], points[0]);
  let path = `M ${start.x.toFixed(2)} ${start.y.toFixed(2)}`;

  for (let index = 0; index < points.length; index += 1) {
    const currentPoint = points[index];
    const nextPoint = points[(index + 1) % points.length];
    const currentMidpoint = midpoint(currentPoint, nextPoint);
    path += ` Q ${currentPoint.x.toFixed(2)} ${currentPoint.y.toFixed(2)} ${currentMidpoint.x.toFixed(2)} ${currentMidpoint.y.toFixed(2)}`;
  }

  path += " Z";
  return path;
}

function smoothDisplacementAt(index) {
  const previousFar = preview.displacementSamples[(index - 2 + SAMPLE_COUNT) % SAMPLE_COUNT] || 0;
  const previousNear = preview.displacementSamples[(index - 1 + SAMPLE_COUNT) % SAMPLE_COUNT] || 0;
  const current = preview.displacementSamples[index] || 0;
  const nextNear = preview.displacementSamples[(index + 1) % SAMPLE_COUNT] || 0;
  const nextFar = preview.displacementSamples[(index + 2) % SAMPLE_COUNT] || 0;

  return (
    previousFar * 0.08 +
    previousNear * 0.22 +
    current * 0.4 +
    nextNear * 0.22 +
    nextFar * 0.08
  );
}

function samplePosition(progress, displacement) {
  const angle = progress * TAU - Math.PI / 2;
  const anchorX = VISUALIZER_CENTER + Math.cos(angle) * PERIMETER_BASE_RADIUS;
  const anchorY = VISUALIZER_CENTER + Math.sin(angle) * PERIMETER_BASE_RADIUS;
  const normalX = Math.cos(angle);
  const normalY = Math.sin(angle);
  const scale = DISPLACEMENT_SCALE * getMoodScale();

  return {
    x: anchorX + normalX * displacement * scale,
    y: anchorY + normalY * displacement * scale,
  };
}

function buildLastRotationPoints(currentIndex) {
  const points = [];

  for (let step = 1; step <= SAMPLE_COUNT; step += 1) {
    const sampleIndex = (currentIndex + step) % SAMPLE_COUNT;
    const sampleProgress = sampleIndex / SAMPLE_COUNT;
    points.push(samplePosition(sampleProgress, smoothDisplacementAt(sampleIndex)));
  }

  return points;
}

function decayHistory(currentIndex, currentDisplacement) {
  const decay = Math.abs(currentDisplacement) < SILENCE_THRESHOLD ? SILENCE_DECAY : BASE_DECAY;

  for (let index = 0; index < SAMPLE_COUNT; index += 1) {
    if (index === currentIndex) {
      continue;
    }
    preview.displacementSamples[index] *= decay;
    if (Math.abs(preview.displacementSamples[index]) < 0.0008) {
      preview.displacementSamples[index] = 0;
    }
  }
}

function renderPerimeter(currentIndex) {
  const path = buildClosedPath(buildLastRotationPoints(currentIndex));
  elements.perimeterShadow.setAttribute("d", path);
  elements.perimeterFill.setAttribute("d", path);
  elements.perimeterHighlight.setAttribute("d", path);
}

function sampleLiveDisplacement() {
  if (!preview.analyser || !preview.timeData) {
    return 0;
  }

  preview.analyser.getFloatTimeDomainData(preview.timeData);

  let signedPeak = 0;
  for (let index = 0; index < preview.timeData.length; index += 1) {
    const sample = preview.timeData[index];
    if (Math.abs(sample) > Math.abs(signedPeak)) {
      signedPeak = sample;
    }
  }

  const gatedPeak = Math.abs(signedPeak) < 0.012 ? 0 : signedPeak;
  const normalized = clamp(gatedPeak * 2.2, -0.55, 0.55);
  preview.smoothedDisplacement = preview.smoothedDisplacement * 0.84 + normalized * 0.16;
  return preview.smoothedDisplacement;
}

function sampleDemoDisplacement(timestamp) {
  const demoDisplacement = Math.sin(timestamp / 180) * 0.28;
  preview.smoothedDisplacement = preview.smoothedDisplacement * 0.86 + demoDisplacement * 0.14;
  return preview.smoothedDisplacement;
}

function updateVisualizerFrame(timestamp, displacementSampler) {
  const progress = (timestamp % ORBIT_PERIOD_MS) / ORBIT_PERIOD_MS;
  const currentIndex = Math.floor(progress * SAMPLE_COUNT) % SAMPLE_COUNT;
  const currentDisplacement = displacementSampler(timestamp);
  preview.displacementSamples[currentIndex] = currentDisplacement;
  decayHistory(currentIndex, currentDisplacement);
  renderPerimeter(currentIndex);
}

function stopPreviewAnimation() {
  if (preview.animationId) {
    cancelAnimationFrame(preview.animationId);
    preview.animationId = 0;
  }
}

function runDemoAnimation() {
  stopPreviewAnimation();

  const loop = (timestamp) => {
    updateVisualizerFrame(timestamp, sampleDemoDisplacement);
    preview.animationId = requestAnimationFrame(loop);
  };

  preview.animationId = requestAnimationFrame(loop);
}

function runLivePreviewAnimation() {
  if (!preview.analyser || !preview.timeData) {
    runDemoAnimation();
    return;
  }

  stopPreviewAnimation();

  const loop = (timestamp) => {
    updateVisualizerFrame(timestamp, sampleLiveDisplacement);
    preview.animationId = requestAnimationFrame(loop);
  };

  preview.animationId = requestAnimationFrame(loop);
}

function getApiBase() {
  return elements.apiBaseInput.value.trim().replace(/\/$/, "");
}

function buildApiUrl(path) {
  const base = getApiBase();
  if (!base) {
    return path;
  }
  return new URL(path, `${base}/`).toString();
}

function buildWebSocketUrl(path) {
  const base = getApiBase();
  const url = base
    ? new URL(path, `${base}/`)
    : new URL(path, window.location.href);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}

function saveApiBase() {
  localStorage.setItem("dnd-api-base-url", getApiBase());
}

async function api(path, options = {}) {
  const response = await fetch(buildApiUrl(path), {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  let payload = {};
  try {
    payload = await response.json();
  } catch (_error) {
    payload = {};
  }

  if (!response.ok) {
    throw new Error(payload.detail || payload.error || "Request failed.");
  }
  return payload;
}

async function apiMultipart(path, body) {
  const response = await fetch(buildApiUrl(path), {
    method: "POST",
    body,
  });

  let payload = {};
  try {
    payload = await response.json();
  } catch (_error) {
    payload = {};
  }

  if (!response.ok) {
    throw new Error(payload.detail || payload.error || "Upload failed.");
  }
  return payload;
}

function formatTime(timestamp) {
  if (!timestamp) {
    return "No timestamp";
  }
  try {
    return new Date(timestamp * 1000).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch (_error) {
    return "No timestamp";
  }
}

function getMusicState() {
  return appState.runtime.music || appState.session.music || {};
}

function getMusicEngineLabel() {
  const music = getMusicState();
  if (!appState.runtime.config?.music_enabled) {
    return "Off";
  }
  if (!music.available) {
    return "Unavailable";
  }
  if (player.status === "streaming") {
    return "Live";
  }
  if (player.status === "connecting" || music.connected) {
    return "Arming";
  }
  return "Idle";
}

function buildStatusMessage() {
  if (appState.error) {
    return appState.error;
  }

  const runtimeMessage = appState.runtime.running
    ? "Browser audio streaming is active."
    : "No live transcription is running.";

  const previewMessage = {
    granted: "Browser microphone capture is active.",
    denied: "Browser microphone access was denied. Visualizer is in demo mode.",
    unsupported: "Browser audio capture is unavailable here. Visualizer is in demo mode.",
    idle: "Waiting to open the browser microphone.",
  }[preview.permission] || "Visualizer is in demo mode.";

  const processingMessage = appState.runtime.processing
    ? "Backend is processing a chunk."
    : "Backend is idle.";

  const music = getMusicState();
  let musicMessage = "Music is disabled.";
  if (appState.runtime.config?.music_enabled) {
    if (!music.available) {
      musicMessage = music.error ? `Music unavailable: ${music.error}` : "Music engine unavailable.";
    } else if (player.status === "streaming") {
      musicMessage = "Music stream is live in the browser.";
    } else if (player.status === "connecting") {
      musicMessage = "Music stream is connecting.";
    } else {
      musicMessage = "Music engine is ready.";
    }
  }

  return `${runtimeMessage} ${previewMessage} ${processingMessage} ${musicMessage}`;
}

function renderTags() {
  const tags = [];
  tags.push(titleCase(appState.session.current_mood || "default"));
  tags.push(appState.runtime.running ? "Browser Live" : "Snapshot");
  tags.push(preview.permission === "granted" ? "Mic Open" : "Demo Visuals");
  if (appState.runtime.config?.music_enabled) {
    tags.push(`Music ${titleCase(appState.runtime.config.music_mode || "raw")}`);
  }

  elements.tags.innerHTML = tags.map((tag) => `<span class="tag">${tag}</span>`).join("");
}

function renderMoodGrid() {
  const allowedMoods = appState.session.allowed_moods || [];
  const currentMood = appState.session.current_mood || "default";
  elements.moodGrid.innerHTML = allowedMoods
    .map((mood) => {
      const activeClass = mood === currentMood ? "mood-chip active" : "mood-chip";
      return `<span class="${activeClass}">${titleCase(mood)}</span>`;
    })
    .join("");
}

function renderUtterances() {
  const utterances = [...(appState.session.utterances || [])].slice(-6).reverse();
  if (!utterances.length) {
    elements.utteranceList.innerHTML = `<article class="utterance-card"><time>Waiting</time><p>Start a live session to populate the scene feed.</p></article>`;
    return;
  }

  elements.utteranceList.innerHTML = utterances
    .map(
      (utterance) => `
        <article class="utterance-card">
          <time>${formatTime(utterance.timestamp)}</time>
          <p>${utterance.text}</p>
        </article>
      `,
    )
    .join("");
}

function renderTranscript() {
  const transcript = appState.session.transcript || appState.session.partial_text || "";
  elements.transcriptMeta.textContent = appState.runtime.running
    ? "Uploading browser audio to FastAPI while the soundtrack streams back over WebSocket."
    : "Showing the most recent saved session snapshot.";
  elements.transcriptBox.textContent = transcript || "No transcript available yet.";
}

function renderMusicPanel() {
  const music = getMusicState();
  const modeLabel = titleCase(music.mode || appState.runtime.config?.music_mode || "raw");
  elements.musicStatusCopy.textContent = appState.runtime.config?.music_enabled
    ? `${modeLabel} music mode. ${music.connected ? "Backend generator is connected." : "Waiting for the generator."}`
    : "Music generation is disabled for this session.";
  elements.musicPrompt.textContent = music.prompt || "No music prompt yet.";
  elements.musicReason.textContent = music.last_reason || "The backend has not steered the soundtrack yet.";
  elements.musicPalette.textContent = music.palette_summary || "No palette";
}

function renderLogs() {
  const logs = appState.runtime.logs || [];
  elements.logBox.textContent = logs.length ? logs.join("\n") : "No logs yet.";
}

function updateControls() {
  const config = appState.runtime.config || {};
  if (!appState.runtime.running) {
    elements.languageInput.value = config.language || "en";
    elements.chunkSecondsInput.value = String(config.chunk_seconds || 2.5);
    elements.musicModeSelect.value = config.music_mode || "raw";
  }

  elements.languageInput.disabled = appState.runtime.running;
  elements.chunkSecondsInput.disabled = appState.runtime.running;
  elements.moodAgentToggle.disabled = appState.runtime.running;
  elements.musicToggle.disabled = appState.runtime.running;
  elements.musicModeSelect.disabled = appState.runtime.running || elements.musicToggle.getAttribute("aria-pressed") !== "true";

  const moodAgentEnabled = Boolean(config.mood_agent);
  elements.moodAgentToggle.classList.toggle("active", moodAgentEnabled);
  elements.moodAgentToggle.textContent = moodAgentEnabled ? "Enabled" : "Disabled";
  elements.moodAgentToggle.setAttribute("aria-pressed", String(moodAgentEnabled));

  const musicEnabled = Boolean(config.music_enabled);
  elements.musicToggle.classList.toggle("active", musicEnabled);
  elements.musicToggle.textContent = musicEnabled ? "Enabled" : "Disabled";
  elements.musicToggle.setAttribute("aria-pressed", String(musicEnabled));

  elements.enginePill.textContent = appState.runtime.running ? "Live" : titleCase(appState.runtime.mode || "idle");
  elements.micButtonIcon.textContent = appState.runtime.running ? "STOP" : "LIVE";
  elements.micButtonLabel.textContent = appState.runtime.running ? "Stop" : "Start";
  elements.tooltip.textContent = appState.runtime.running ? "Stop browser stream" : "Start browser stream";
  elements.sessionToggle.setAttribute(
    "aria-label",
    appState.runtime.running ? "Stop live session" : "Start live session",
  );
}

function updateCopy() {
  const utteranceCount = appState.session.utterance_count || 0;
  const latestMoodEvent = appState.session.latest_mood_event;

  elements.shell.dataset.mood = appState.session.current_mood || "default";
  elements.subtitle.textContent = appState.runtime.running
    ? `Streaming browser audio to the API. ${utteranceCount} utterance${utteranceCount === 1 ? "" : "s"} captured so far.`
    : utteranceCount
      ? "Displaying the most recent saved scene snapshot."
      : "Standing by for a live session.";
  elements.statusBanner.textContent = buildStatusMessage();
  elements.metricMood.textContent = titleCase(appState.session.current_mood || "default");
  elements.metricConfidence.textContent = `${Math.round((latestMoodEvent?.confidence || 0) * 100)}%`;
  elements.metricUtterances.textContent = String(utteranceCount);
  elements.metricMusic.textContent = getMusicEngineLabel();
  elements.moodReason.textContent = latestMoodEvent?.reason || "No mood classification yet.";
}

function applyState(payload) {
  appState.runtime = payload.runtime || appState.runtime;
  appState.session = payload.session || appState.session;
  appState.error = "";

  updateCopy();
  updateControls();
  renderTags();
  renderMoodGrid();
  renderUtterances();
  renderTranscript();
  renderMusicPanel();
  renderLogs();
}

async function refreshState() {
  try {
    const payload = await api("/api/state");
    applyState(payload);
  } catch (error) {
    appState.error = error.message || "Could not reach the API.";
    updateCopy();
  }
}

function getChunkSeconds() {
  const parsed = Number.parseFloat(elements.chunkSecondsInput.value);
  if (!Number.isFinite(parsed)) {
    return 2.5;
  }
  return clamp(parsed, 1, 10);
}

function getMusicVolume() {
  const parsed = Number.parseFloat(elements.musicVolumeInput.value);
  if (!Number.isFinite(parsed)) {
    return 0.68;
  }
  return clamp(parsed / 100, 0, 1);
}

function mergeBuffers(buffers) {
  const totalLength = buffers.reduce((sum, buffer) => sum + buffer.length, 0);
  const merged = new Float32Array(totalLength);
  let offset = 0;
  buffers.forEach((buffer) => {
    merged.set(buffer, offset);
    offset += buffer.length;
  });
  return merged;
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const writeString = (offset, string) => {
    for (let index = 0; index < string.length; index += 1) {
      view.setUint8(offset + index, string.charCodeAt(index));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  samples.forEach((sample) => {
    const value = clamp(sample, -1, 1);
    view.setInt16(offset, value < 0 ? value * 0x8000 : value * 0x7fff, true);
    offset += 2;
  });

  return new Blob([buffer], { type: "audio/wav" });
}

async function ensureAudioCapture() {
  if (preview.permission === "granted" && preview.stream && preview.context) {
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    preview.permission = "unsupported";
    preview.mode = "demo";
    runDemoAnimation();
    throw new Error("Browser audio capture is not supported here.");
  }

  try {
    preview.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    preview.context = new AudioContext();
    preview.source = preview.context.createMediaStreamSource(preview.stream);
    preview.analyser = preview.context.createAnalyser();
    preview.analyser.fftSize = 1024;
    preview.timeData = new Float32Array(preview.analyser.fftSize);
    preview.processor = preview.context.createScriptProcessor(4096, 1, 1);
    preview.source.connect(preview.analyser);
    preview.source.connect(preview.processor);
    preview.processor.connect(preview.context.destination);
    preview.processor.onaudioprocess = (event) => {
      if (!preview.activeSessionId || preview.stopping) {
        return;
      }
      const channelData = event.inputBuffer.getChannelData(0);
      preview.chunkBuffers.push(new Float32Array(channelData));
    };
    preview.permission = "granted";
    preview.mode = "live";
    preview.smoothedDisplacement = 0;
    preview.displacementSamples.fill(0);
    runLivePreviewAnimation();
  } catch (_error) {
    preview.permission = "denied";
    preview.mode = "demo";
    runDemoAnimation();
    throw new Error("Microphone access was denied in the browser.");
  }
}

async function flushChunkBuffer() {
  if (!preview.activeSessionId) {
    preview.chunkBuffers = [];
    return;
  }
  if (!preview.chunkBuffers.length || !preview.context) {
    return;
  }

  const merged = mergeBuffers(preview.chunkBuffers);
  preview.chunkBuffers = [];
  if (!merged.length) {
    return;
  }

  preview.chunkSequence += 1;
  preview.uploadQueue.push({
    sessionId: preview.activeSessionId,
    sequence: preview.chunkSequence,
    blob: encodeWav(merged, preview.context.sampleRate),
  });
  await processUploadQueue();
}

async function processUploadQueue() {
  if (preview.uploadInFlight) {
    return;
  }
  preview.uploadInFlight = true;

  while (preview.uploadQueue.length) {
    const item = preview.uploadQueue.shift();
    if (!item) {
      continue;
    }

    const formData = new FormData();
    formData.append("file", item.blob, `chunk-${String(item.sequence).padStart(4, "0")}.wav`);
    formData.append("sequence", String(item.sequence));

    try {
      const payload = await apiMultipart(`/api/session/${item.sessionId}/chunk`, formData);
      applyState(payload);
    } catch (error) {
      appState.error = error.message || "Chunk upload failed.";
      updateCopy();
      break;
    }
  }

  preview.uploadInFlight = false;
}

async function waitForUploadsToDrain() {
  while (preview.uploadInFlight || preview.uploadQueue.length) {
    await new Promise((resolve) => window.setTimeout(resolve, 150));
  }
}

function startFlushTimer() {
  window.clearInterval(preview.flushTimerId);
  preview.flushTimerId = window.setInterval(() => {
    flushChunkBuffer().catch((error) => {
      appState.error = error.message || "Failed to flush audio chunk.";
      updateCopy();
    });
  }, getChunkSeconds() * 1000);
}

async function cleanupAudioCapture() {
  window.clearInterval(preview.flushTimerId);
  preview.flushTimerId = 0;
  preview.chunkBuffers = [];
  preview.chunkSequence = 0;
  preview.activeSessionId = "";

  if (preview.processor) {
    preview.processor.disconnect();
    preview.processor.onaudioprocess = null;
    preview.processor = null;
  }
  if (preview.source) {
    preview.source.disconnect();
    preview.source = null;
  }
  if (preview.stream) {
    preview.stream.getTracks().forEach((track) => track.stop());
    preview.stream = null;
  }
  if (preview.context) {
    await preview.context.close();
    preview.context = null;
  }

  preview.analyser = null;
  preview.timeData = null;
  preview.permission = "idle";
  preview.mode = "demo";
  preview.stopping = false;
  preview.smoothedDisplacement = 0;
  preview.displacementSamples.fill(0);
  runDemoAnimation();
}

function resetMusicBuffers() {
  player.queue = [];
  player.currentChunk = null;
  player.currentFrameOffset = 0;
  player.bufferedFrames = 0;
}

function fillPlaybackBuffer(outputBuffer) {
  const left = outputBuffer.getChannelData(0);
  const right = outputBuffer.getChannelData(1);
  left.fill(0);
  right.fill(0);

  let frameIndex = 0;
  while (frameIndex < left.length) {
    if (!player.currentChunk) {
      const nextChunk = player.queue.shift();
      if (!nextChunk) {
        break;
      }
      player.currentChunk = nextChunk;
      player.currentFrameOffset = 0;
    }

    const totalFrames = Math.floor(player.currentChunk.length / player.channels);
    const remainingFrames = totalFrames - player.currentFrameOffset;
    const framesToCopy = Math.min(left.length - frameIndex, remainingFrames);

    for (let index = 0; index < framesToCopy; index += 1) {
      const sourceIndex = (player.currentFrameOffset + index) * player.channels;
      left[frameIndex + index] = player.currentChunk[sourceIndex] / 32768;
      right[frameIndex + index] = player.currentChunk[sourceIndex + 1] / 32768;
    }

    frameIndex += framesToCopy;
    player.currentFrameOffset += framesToCopy;
    player.bufferedFrames = Math.max(0, player.bufferedFrames - framesToCopy);

    if (player.currentFrameOffset >= totalFrames) {
      player.currentChunk = null;
      player.currentFrameOffset = 0;
    }
  }
}

function updateVolume() {
  if (player.gainNode) {
    player.gainNode.gain.value = getMusicVolume();
  }
}

async function ensureMusicPlayback() {
  if (player.context) {
    if (player.context.state === "suspended") {
      await player.context.resume();
    }
    updateVolume();
    return;
  }

  const preferredRate = getMusicState().sample_rate || 48000;
  try {
    player.context = new AudioContext({ sampleRate: preferredRate });
  } catch (_error) {
    player.context = new AudioContext();
  }
  player.sampleRate = preferredRate;
  player.channels = 2;
  player.gainNode = player.context.createGain();
  try {
    player.processor = player.context.createScriptProcessor(4096, 0, 2);
  } catch (_error) {
    player.processor = player.context.createScriptProcessor(4096, 1, 2);
  }
  player.processor.onaudioprocess = (event) => {
    fillPlaybackBuffer(event.outputBuffer);
  };
  player.processor.connect(player.gainNode);
  player.gainNode.connect(player.context.destination);
  updateVolume();
  await player.context.resume();
}

function enqueueMusicChunk(arrayBuffer) {
  const chunk = new Int16Array(arrayBuffer);
  if (!chunk.length) {
    return;
  }

  player.queue.push(chunk);
  player.bufferedFrames += Math.floor(chunk.length / player.channels);

  const maxFrames = player.sampleRate * MUSIC_BUFFER_MAX_SECONDS;
  while (player.bufferedFrames > maxFrames && player.queue.length > 1) {
    const dropped = player.queue.shift();
    if (!dropped) {
      break;
    }
    player.bufferedFrames -= Math.floor(dropped.length / player.channels);
  }
}

async function closeMusicSocket() {
  if (!player.websocket) {
    return;
  }
  player.manualClose = true;
  player.websocket.close();
  player.websocket = null;
}

async function cleanupMusicPlayback() {
  await closeMusicSocket();
  resetMusicBuffers();
  if (player.processor) {
    player.processor.disconnect();
    player.processor.onaudioprocess = null;
    player.processor = null;
  }
  if (player.gainNode) {
    player.gainNode.disconnect();
    player.gainNode = null;
  }
  if (player.context) {
    await player.context.close();
    player.context = null;
  }
  player.status = "idle";
  player.manualClose = false;
}

async function startMusicStream(sessionId) {
  const music = getMusicState();
  if (!appState.runtime.config?.music_enabled || !music.available || !sessionId) {
    return;
  }

  await ensureMusicPlayback();
  await closeMusicSocket();
  resetMusicBuffers();

  player.status = "connecting";
  player.manualClose = false;
  const socket = new WebSocket(buildWebSocketUrl(`/api/session/${sessionId}/music/stream`));
  socket.binaryType = "arraybuffer";
  player.websocket = socket;

  socket.addEventListener("message", (event) => {
    if (typeof event.data === "string") {
      let payload = {};
      try {
        payload = JSON.parse(event.data);
      } catch (_error) {
        payload = {};
      }
      if (payload.type === "init" && payload.music) {
        appState.runtime.music = payload.music;
        player.sampleRate = payload.music.sample_rate || player.sampleRate;
        player.channels = payload.music.channels || 2;
        player.status = "streaming";
        updateCopy();
        renderMusicPanel();
      } else if (payload.type === "error") {
        appState.error = payload.detail || "Music stream failed.";
        updateCopy();
      }
      return;
    }

    enqueueMusicChunk(event.data);
    if (player.status !== "streaming") {
      player.status = "streaming";
      updateCopy();
    }
  });

  socket.addEventListener("close", () => {
    if (player.websocket === socket) {
      player.websocket = null;
    }
    if (!player.manualClose) {
      player.status = "idle";
      updateCopy();
    }
    player.manualClose = false;
  });

  socket.addEventListener("error", () => {
    appState.error = "Music websocket failed.";
    updateCopy();
  });
}

async function startSession() {
  await ensureAudioCapture();
  const payload = await api("/api/session/start", {
    method: "POST",
    body: JSON.stringify({
      language: elements.languageInput.value.trim() || "en",
      chunk_seconds: getChunkSeconds(),
      mood_agent: elements.moodAgentToggle.getAttribute("aria-pressed") === "true",
      music_enabled: elements.musicToggle.getAttribute("aria-pressed") === "true",
      music_mode: elements.musicModeSelect.value,
    }),
  });
  applyState(payload);
  preview.activeSessionId = payload.runtime.current_session_id || "";
  startFlushTimer();
  if (payload.runtime.config?.music_enabled && payload.runtime.music?.available) {
    await startMusicStream(preview.activeSessionId);
  }
}

async function stopSession() {
  preview.stopping = true;
  await flushChunkBuffer();
  await waitForUploadsToDrain();

  const sessionId = appState.runtime.current_session_id || preview.activeSessionId;
  if (sessionId) {
    const payload = await api(`/api/session/${sessionId}/stop`, {
      method: "POST",
      body: JSON.stringify({}),
    });
    applyState(payload);
  }

  await cleanupMusicPlayback();
  await cleanupAudioCapture();
}

function onToggleMoodAgent() {
  if (appState.runtime.running) {
    return;
  }
  const nextValue = elements.moodAgentToggle.getAttribute("aria-pressed") !== "true";
  elements.moodAgentToggle.setAttribute("aria-pressed", String(nextValue));
  elements.moodAgentToggle.classList.toggle("active", nextValue);
  elements.moodAgentToggle.textContent = nextValue ? "Enabled" : "Disabled";
}

function onToggleMusic() {
  if (appState.runtime.running) {
    return;
  }
  const nextValue = elements.musicToggle.getAttribute("aria-pressed") !== "true";
  elements.musicToggle.setAttribute("aria-pressed", String(nextValue));
  elements.musicToggle.classList.toggle("active", nextValue);
  elements.musicToggle.textContent = nextValue ? "Enabled" : "Disabled";
  elements.musicModeSelect.disabled = !nextValue;
}

async function onSessionToggle() {
  try {
    appState.error = "";
    if (appState.runtime.running) {
      await stopSession();
    } else {
      await startSession();
    }
  } catch (error) {
    appState.error = error.message || "Session command failed.";
    updateCopy();
  }
}

function attachEvents() {
  elements.refreshButton.addEventListener("click", refreshState);
  elements.moodAgentToggle.addEventListener("click", onToggleMoodAgent);
  elements.musicToggle.addEventListener("click", onToggleMusic);
  elements.musicVolumeInput.addEventListener("input", updateVolume);
  elements.sessionToggle.addEventListener("click", onSessionToggle);
  elements.apiBaseInput.addEventListener("change", () => {
    saveApiBase();
    refreshState();
  });
}

function init() {
  const savedApiBase = localStorage.getItem("dnd-api-base-url");
  if (savedApiBase) {
    elements.apiBaseInput.value = savedApiBase;
  }

  runDemoAnimation();
  attachEvents();
  updateVolume();
  refreshState();
  window.setInterval(refreshState, POLL_INTERVAL_MS);
}

init();

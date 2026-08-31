/**
 * Auto Viral Cuts - Client-Side WebCodecs & Canvas 2D Engine
 * Decodes video locally, crops to 9:16, renders dynamic Hormozi subtitles,
 * and encodes final MP4 on the user's GPU via WebCodecs + mp4-muxer.
 */

import { Muxer, ArrayBufferTarget } from "./vendor/mp4-muxer.js";
import { ClientAudioExtractor } from "./audio_extractor.js";

// DOM Elements
const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const fileInfoBox = document.getElementById("fileInfoBox");
const fileNameVal = document.getElementById("fileNameVal");
const fileSizeVal = document.getElementById("fileSizeVal");
const fileDurationVal = document.getElementById("fileDurationVal");
const fileResVal = document.getElementById("fileResVal");

const btnGenerateManifest = document.getElementById("btnGenerateManifest");
const statusBox = document.getElementById("statusBox");
const statusText = document.getElementById("statusText");
const progressBarFill = document.getElementById("progressBarFill");

const cutsListContainer = document.getElementById("cutsListContainer");
const previewCanvas = document.getElementById("previewCanvas");
const canvasPlaceholder = document.getElementById("canvasPlaceholder");
const ctx = previewCanvas.getContext("2d");

const btnPlayPause = document.getElementById("btnPlayPause");
const seekBar = document.getElementById("seekBar");
const timeDisplay = document.getElementById("timeDisplay");

const btnExport = document.getElementById("btnExport");
const resolutionSelect = document.getElementById("resolutionSelect");
const exportProgressArea = document.getElementById("exportProgressArea");
const exportStatusLabel = document.getElementById("exportStatusLabel");
const exportProgressFill = document.getElementById("exportProgressFill");
const webcodecsBadge = document.getElementById("webcodecsBadge");

// Hidden video element for local decoding & playback
const video = document.createElement("video");
video.playsInline = true;
video.muted = false;

// Internal State
let originalFile = null;
let cutsManifests = [];
let activeCut = null;
let isPlaying = false;
let animFrameId = null;
let groupedPhrases = []; // Cached word phrases for active cut

// 1. Initial Capability Check
function checkCapabilities() {
  if (typeof VideoEncoder !== "undefined" && typeof VideoDecoder !== "undefined") {
    webcodecsBadge.textContent = "WebCodecs GPU Ativo";
    webcodecsBadge.className = "badge badge-webcodecs";
  } else {
    webcodecsBadge.textContent = "WebCodecs Indisponível";
    webcodecsBadge.className = "badge";
    webcodecsBadge.style.background = "rgba(244, 63, 94, 0.2)";
    webcodecsBadge.style.color = "#f43f5e";
  }

}

// 2. File Selection & Drag-and-Drop
function initDropzone() {
  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });
}

function handleFileSelected(file) {
  originalFile = file;
  fileNameVal.textContent = file.name;
  fileSizeVal.textContent = `${(file.size / (1024 * 1024)).toFixed(1)} MB`;

  const videoUrl = URL.createObjectURL(file);
  video.src = videoUrl;

  video.onloadedmetadata = () => {
    fileDurationVal.textContent = formatTime(video.duration);
    fileResVal.textContent = `${video.videoWidth}x${video.videoHeight}`;
    fileInfoBox.style.display = "block";
    btnGenerateManifest.disabled = false;

    // Set internal canvas resolution to 1080x1920 (standard 9:16)
    previewCanvas.width = 1080;
    previewCanvas.height = 1920;
    canvasPlaceholder.style.display = "none";

    // Draw initial centered frame
    video.currentTime = 0;
  };
}

video.onseeked = () => {
  if (!isPlaying) {
    renderCurrentFrame();
  }
};

// 3. Manifest Generation (Local Audio Extraction -> Backend IA)
btnGenerateManifest.addEventListener("click", async () => {
  if (!originalFile) return;

  btnGenerateManifest.disabled = true;
  statusBox.style.display = "block";
  updateProgress("Preparando extração...", 5);

  try {
    // Step A: Extract lightweight audio locally in browser
    const audioBlob = await ClientAudioExtractor.extractAudioBlob(
      originalFile,
      (text, pct) => updateProgress(text, pct * 0.4)
    );

    // Step B: Send audio to /api/v1/generate-manifest
    const aiProvider = document.getElementById("aiProviderSelect")
      ? document.getElementById("aiProviderSelect").value
      : "groq";
    const langSelect = document.getElementById("languageModeSelect");
    const subLang = langSelect ? langSelect.value : "original";
    const groqKeyInput = document.getElementById("groqApiKeyInput");
    const groqKey = groqKeyInput ? groqKeyInput.value.trim() : "";

    const providerLabel = aiProvider === "groq" ? "Groq LPU (LLaMA 3.3 70B)" : "Google Gemini 3.6";
    updateProgress(`Transcrevendo e analisando texto via ${providerLabel}...`, 50);

    const formData = new FormData();
    formData.append("file", audioBlob, "extracted_audio.wav");
    formData.append("max_clips", document.getElementById("maxClipsSelect").value);
    formData.append("crop_mode", "center_crop");
    formData.append("ai_provider", aiProvider);
    formData.append("subtitle_language", subLang);
    formData.append("translate_to_pt", subLang === "pt_br" ? "true" : "false");

    // Parse duration range from selector (format: "30-60")
    const durationSelect = document.getElementById("clipDurationSelect");
    if (durationSelect && durationSelect.value) {
      const [minDur, maxDur] = durationSelect.value.split("-").map(Number);
      if (!isNaN(minDur) && !isNaN(maxDur)) {
        formData.append("min_duration_seconds", minDur);
        formData.append("max_duration_seconds", maxDur);
      }
    }

    if (groqKey) {
      formData.append("groq_api_key", groqKey);
    }

    const promptInput = document.getElementById("customPromptInput").value.trim();
    if (promptInput) {
      formData.append("custom_prompt", promptInput);
    }


    const response = await fetch("/api/v1/generate-manifest", {
      method: "POST",
      body: formData,
    });


    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: "Erro desconhecido" }));
      throw new Error(err.detail || `Erro HTTP ${response.status}`);
    }

    updateProgress("Processando manifesto estruturado de cortes...", 90);
    cutsManifests = await response.json();

    if (!cutsManifests || cutsManifests.length === 0) {
      throw new Error("Nenhum corte viral foi identificado para este vídeo.");
    }

    updateProgress(`Sucesso! ${cutsManifests.length} cortes virais gerados.`, 100);
    renderCutsList(cutsManifests);

    // Select first cut by default
    selectCut(cutsManifests[0]);
  } catch (err) {
    console.error("Erro na geração de manifesto:", err);
    updateProgress(`Erro: ${err.message}`, 100);
    statusBox.style.borderLeftColor = "var(--accent-rose)";
  } finally {
    btnGenerateManifest.disabled = false;
  }
});

function updateProgress(message, percentage) {
  statusText.textContent = message;
  progressBarFill.style.width = `${Math.min(100, Math.max(0, percentage))}%`;
}

// 4. Render Cuts List in Sidebar
function renderCutsList(manifests) {
  cutsListContainer.innerHTML = "";

  manifests.forEach((cut, idx) => {
    const duration = (cut.end_sec - cut.start_sec).toFixed(1);
    const item = document.createElement("div");
    item.className = `cut-item ${idx === 0 ? "active" : ""}`;
    item.id = `cut_item_${cut.cut_id}`;

    item.innerHTML = `
      <div class="cut-header">
        <div class="cut-title" title="${cut.title}">${cut.title}</div>
        <span class="score-badge">Score: ${cut.viral_score}</span>
      </div>
      <div class="cut-details">
        <span>${formatTime(cut.start_sec)} - ${formatTime(cut.end_sec)} (${duration}s)</span>
        <span>${cut.words ? cut.words.length : 0} palavras</span>
      </div>
    `;


    item.addEventListener("click", () => {
      document.querySelectorAll(".cut-item").forEach((el) => el.classList.remove("active"));
      item.classList.add("active");
      selectCut(cut);
    });

    cutsListContainer.appendChild(item);
  });
}

function convertCuesToPhrases(cues, cutStartSec) {
  if (!cues || cues.length === 0) return [];
  return cues.map((cue) => {
    const start_ms = Math.max(0, Math.round((cue.start - cutStartSec) * 1000));
    const end_ms = Math.max(start_ms + 100, Math.round((cue.end - cutStartSec) * 1000));
    const rawWords = (cue.text || "").trim().split(/\s+/).filter(Boolean);
    const duration = end_ms - start_ms;
    const wordDur = duration / (rawWords.length || 1);

    const words = rawWords.map((w, idx) => ({
      word: w,
      start_ms: Math.round(start_ms + idx * wordDur),
      end_ms: Math.round(start_ms + (idx + 1) * wordDur),
    }));

    return {
      start_ms,
      end_ms,
      words,
    };
  });
}

function selectCut(cut) {
  activeCut = cut;
  btnExport.disabled = false;
  document.getElementById("statCutTitle").textContent = cut.title_pt || cut.title;
  document.getElementById("statCutDuration").textContent = `${(cut.end_sec - cut.start_sec).toFixed(1)}s`;
  
  const wordsCount = cut.subtitles_pt && cut.subtitles_pt.length > 0 
    ? cut.subtitles_pt.reduce((acc, c) => acc + (c.text ? c.text.split(/\s+/).length : 0), 0)
    : (cut.words ? cut.words.length : 0);
  document.getElementById("statCutWords").textContent = wordsCount;

  // Use translated cues if available, otherwise original words
  if (cut.subtitles_pt && cut.subtitles_pt.length > 0) {
    groupedPhrases = convertCuesToPhrases(cut.subtitles_pt, cut.start_sec);
  } else {
    groupedPhrases = groupWordsIntoPhrases(cut.words || []);
  }

  // Seek video to start of cut
  video.currentTime = cut.start_sec;
  seekBar.value = 0;
  updateTimeDisplay(0, cut.end_sec - cut.start_sec);
  renderCurrentFrame();
}

/**
 * Groups words into small, punchy phrases (2-4 words) for TikTok/Reels captions.
 */
function groupWordsIntoPhrases(words) {

  if (!words || words.length === 0) return [];
  const phrases = [];
  let currentGroup = [];

  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    currentGroup.push(w);

    const isLast = i === words.length - 1;
    const nextPause = !isLast && words[i + 1].start_ms - w.end_ms > 400;
    const reachedLength = currentGroup.length >= 3;

    if (isLast || nextPause || reachedLength) {
      phrases.push({
        start_ms: currentGroup[0].start_ms,
        end_ms: currentGroup[currentGroup.length - 1].end_ms,
        words: [...currentGroup],
      });
      currentGroup = [];
    }
  }
  return phrases;
}

// 5. Canvas 2D Rendering Engine (9:16 Center Crop + Hormozi Subtitles)

function renderCurrentFrame() {
  if (!video.videoWidth || !video.videoHeight) return;

  const w = previewCanvas.width;
  const h = previewCanvas.height;

  // Clear canvas
  ctx.clearRect(0, 0, w, h);

  // Draw 9:16 Center Crop
  drawCenterCrop(ctx, video, w, h);

  // Draw Dynamic Subtitles
  if (activeCut) {
    const clipRelTimeMs = Math.max(0, (video.currentTime - activeCut.start_sec) * 1000);
    drawDynamicSubtitles(ctx, groupedPhrases, clipRelTimeMs, w, h);
  }
}

/**
 * Draws landscape/any video center-cropped into a vertical 9:16 canvas.
 */
function drawCenterCrop(context, videoSource, targetW, targetH) {
  const srcW = videoSource.videoWidth;
  const srcH = videoSource.videoHeight;
  const targetAspect = targetW / targetH; // 9 / 16 = 0.5625
  const srcAspect = srcW / srcH;

  let cropW, cropH, cropX, cropY;

  if (srcAspect > targetAspect) {
    // Video is wider than 9:16 (standard landscape 16:9)
    cropH = srcH;
    cropW = srcH * targetAspect;
    cropX = (srcW - cropW) / 2;
    cropY = 0;
  } else {
    // Video is taller than 9:16
    cropW = srcW;
    cropH = srcW / targetAspect;
    cropX = 0;
    cropY = (srcH - cropH) / 2;
  }

  context.drawImage(videoSource, cropX, cropY, cropW, cropH, 0, 0, targetW, targetH);
}

/**
 * Draws Hormozi-style viral subtitles:
 * - High-impact uppercase typography
 * - Thick black outline (stroke) + drop shadow
 * - Active spoken word highlighted in vibrant yellow/gold with subtle scale pop
 * - Positioned in the vertical safe-zone (70% height)
 */
function drawDynamicSubtitles(context, phrases, currentClipTimeMs, canvasW, canvasH) {
  if (!phrases || phrases.length === 0) return;

  // Find active phrase
  const activePhrase = phrases.find(
    (p) => currentClipTimeMs >= p.start_ms && currentClipTimeMs <= p.end_ms + 150
  );

  if (!activePhrase) return;

  context.save();

  // Typography settings
  const fontSize = Math.round(canvasW * 0.065); // ~70px on 1080w
  context.font = `900 ${fontSize}px "Impact", "Montserrat", "Inter", sans-serif`;
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.lineJoin = "round";

  // Calculate total phrase width for horizontal centering
  const spaceWidth = context.measureText(" ").width;
  const wordMetrics = activePhrase.words.map((w) => {
    const text = w.word.toUpperCase();
    const width = context.measureText(text).width;
    const isActive = currentClipTimeMs >= w.start_ms && currentClipTimeMs <= w.end_ms;
    return { text, width, isActive };
  });

  const totalWidth =
    wordMetrics.reduce((acc, m) => acc + m.width, 0) + (wordMetrics.length - 1) * spaceWidth;

  let startX = (canvasW - totalWidth) / 2;
  const posY = canvasH * 0.72; // Safe zone (72% vertical)

  // Draw each word in phrase
  wordMetrics.forEach((item) => {
    const wordCenter = startX + item.width / 2;

    context.save();
    if (item.isActive) {
      // Subtle scale-up pop for active spoken word
      context.translate(wordCenter, posY);
      context.scale(1.12, 1.12);
      context.translate(-wordCenter, -posY);
    }

    // Heavy black outline
    context.strokeStyle = "#000000";
    context.lineWidth = Math.round(fontSize * 0.22); // ~15px stroke
    context.strokeText(item.text, wordCenter, posY);

    // Active word = Vibrant Yellow / Inactive = Crisp White
    context.fillStyle = item.isActive ? "#FFE600" : "#FFFFFF";
    context.fillText(item.text, wordCenter, posY);

    context.restore();

    startX += item.width + spaceWidth;
  });

  context.restore();
}

// 6. Playback & Timeline Controls
btnPlayPause.addEventListener("click", () => {
  if (isPlaying) {
    pausePlayback();
  } else {
    startPlayback();
  }
});

function startPlayback() {
  if (!activeCut) return;
  isPlaying = true;
  btnPlayPause.textContent = "Pause";
  video.play().catch(() => {});
  playbackLoop();
}

function pausePlayback() {
  isPlaying = false;
  btnPlayPause.textContent = "Play";
  video.pause();
  if (animFrameId) {
    cancelAnimationFrame(animFrameId);
  }
}


function playbackLoop() {
  if (!isPlaying) return;

  if (activeCut) {
    // Loop playback within cut bounds
    if (video.currentTime >= activeCut.end_sec || video.currentTime < activeCut.start_sec) {
      video.currentTime = activeCut.start_sec;
    }

    const elapsed = video.currentTime - activeCut.start_sec;
    const duration = activeCut.end_sec - activeCut.start_sec;
    seekBar.value = duration > 0 ? (elapsed / duration) * 100 : 0;
    updateTimeDisplay(elapsed, duration);
  }

  renderCurrentFrame();
  animFrameId = requestAnimationFrame(playbackLoop);
}

seekBar.addEventListener("input", (e) => {
  if (!activeCut) return;
  const pct = parseFloat(e.target.value) / 100;
  const duration = activeCut.end_sec - activeCut.start_sec;
  video.currentTime = activeCut.start_sec + pct * duration;
  updateTimeDisplay(pct * duration, duration);
  renderCurrentFrame();
});

function updateTimeDisplay(current, total) {
  timeDisplay.textContent = `${formatTime(current)} / ${formatTime(total)}`;
}

function formatTime(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  const mins = Math.floor(s / 60);
  const secs = s % 60;
  return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
}

// Helper to find supported codec and acceleration configuration
async function findSupportedEncoderConfig(width, height, bitrate) {
  const candidateCodecs = [
    "avc1.42002a", // Baseline Level 4.2 (Maximum compatibility for 1080x1920)
    "avc1.42001f", // Baseline Level 3.1
    "avc1.4d002a", // Main Level 4.2
    "avc1.64002a", // High Level 4.2
    "avc1.420028", // Baseline Level 4.0
    "avc1.640028", // High Level 4.0
    "vp8",
    "vp09.00.10.08",
  ];

  // Try "no-preference" first to prevent hardware creation errors on Linux/GPU
  const accelerationModes = ["no-preference", "prefer-hardware", "prefer-software"];

  for (const hw of accelerationModes) {
    for (const codec of candidateCodecs) {
      const testConfig = {
        codec: codec,
        width: width,
        height: height,
        bitrate: bitrate,
        framerate: 30,
        hardwareAcceleration: hw,
      };

      try {
        if (typeof VideoEncoder.isConfigSupported === "function") {
          const support = await VideoEncoder.isConfigSupported(testConfig);
          if (support && support.supported) {
            console.log("Configuracao WebCodecs suportada detectada:", support.config || testConfig);
            return support.config || testConfig;
          }
        } else {
          return testConfig;
        }
      } catch (e) {
        // try next combination
      }
    }
  }

  // Safe universal fallback
  return {
    codec: "avc1.42002a",
    width: width,
    height: height,
    bitrate: bitrate,
    framerate: 30,
    hardwareAcceleration: "no-preference",
  };
}

// 7. WebCodecs VideoEncoder + mp4-muxer GPU Export Pipeline
btnExport.addEventListener("click", async () => {
  if (!activeCut || !originalFile) return;

  pausePlayback();
  btnExport.disabled = true;
  exportProgressArea.style.display = "block";

  const is1080p = resolutionSelect.value === "1080x1920";
  const exportW = is1080p ? 1080 : 720;
  const exportH = is1080p ? 1920 : 1280;
  const bitrate = is1080p ? 5_000_000 : 3_000_000;

  exportStatusLabel.innerHTML = `<span>Inicializando WebCodecs VideoEncoder...</span><span>0%</span>`;
  exportProgressFill.style.width = "0%";

  try {
    if (typeof VideoEncoder === "undefined") {
      throw new Error(
        "WebCodecs VideoEncoder não é suportado pelo seu navegador. Use Chrome, Edge ou Safari recente."
      );
    }

    const encoderConfig = await findSupportedEncoderConfig(exportW, exportH, bitrate);
    console.log("Iniciando VideoEncoder com:", encoderConfig);

    // Determine muxer codec from encoder codec
    const isAvc = encoderConfig.codec.startsWith("avc1");
    const isVp9 = encoderConfig.codec.startsWith("vp09") || encoderConfig.codec.startsWith("vp9");
    const isVp8 = encoderConfig.codec.startsWith("vp8");
    const muxerCodec = isAvc ? "avc" : (isVp9 ? "vp9" : (isVp8 ? "vp8" : "avc"));

    // Initialize mp4-muxer
    const muxer = new Muxer({
      target: new ArrayBufferTarget(),
      video: {
        codec: muxerCodec,
        width: exportW,
        height: exportH,
      },
      fastStart: "in-memory",
    });

    let encoderError = null;
    let encoder = new VideoEncoder({
      output: (chunk, meta) => muxer.addVideoChunk(chunk, meta),
      error: (err) => {
        console.error("Erro no VideoEncoder:", err);
        encoderError = err;
      },
    });

    try {
      encoder.configure(encoderConfig);
    } catch (cfgErr) {
      console.warn("Falha ao configurar encoder preferido, tentando fallback Baseline:", cfgErr);
      encoderConfig.codec = "avc1.42001f";
      encoderConfig.hardwareAcceleration = "no-preference";
      encoder.configure(encoderConfig);
    }


    // Create offscreen canvas for rendering export frames
    const offscreen = document.createElement("canvas");
    offscreen.width = exportW;
    offscreen.height = exportH;
    const offCtx = offscreen.getContext("2d", { willReadFrequently: false });

    const fps = 30;
    const startSec = activeCut.start_sec;
    const endSec = activeCut.end_sec;
    const totalFrames = Math.max(1, Math.floor((endSec - startSec) * fps));
    const frameDurationMicros = Math.round(1_000_000 / fps);

    // Frame-by-frame encoding loop
    for (let f = 0; f < totalFrames; f++) {
      if (encoderError) throw encoderError;

      const frameTimeSec = startSec + f / fps;
      video.currentTime = frameTimeSec;

      // Wait until frame is decoded by video element (with safety timeout)
      await new Promise((resolve) => {
        let timer = null;
        const onSeeked = () => {
          if (timer) clearTimeout(timer);
          video.removeEventListener("seeked", onSeeked);
          resolve();
        };
        timer = setTimeout(onSeeked, 200);
        video.addEventListener("seeked", onSeeked, { once: true });
      });

      // Render frame to offscreen canvas
      offCtx.clearRect(0, 0, exportW, exportH);
      drawCenterCrop(offCtx, video, exportW, exportH);
      const relTimeMs = (frameTimeSec - startSec) * 1000;
      drawDynamicSubtitles(offCtx, groupedPhrases, relTimeMs, exportW, exportH);

      // Create VideoFrame and encode
      const videoFrame = new VideoFrame(offscreen, {
        timestamp: f * frameDurationMicros,
        duration: frameDurationMicros,
      });

      const isKeyFrame = f % 60 === 0;
      encoder.encode(videoFrame, { keyFrame: isKeyFrame });
      videoFrame.close();

      // Update progress UI
      const pct = Math.round(((f + 1) / totalFrames) * 100);
      exportStatusLabel.innerHTML = `<span>Renderizando frame ${f + 1}/${totalFrames} na GPU...</span><span>${pct}%</span>`;
      exportProgressFill.style.width = `${pct}%`;
    }

    // Flush encoder and finalize muxer
    exportStatusLabel.innerHTML = `<span>Finalizando container MP4...</span><span>98%</span>`;
    await encoder.flush();
    encoder.close();

    muxer.finalize();
    const mp4Buffer = muxer.target.buffer;
    const mp4Blob = new Blob([mp4Buffer], { type: "video/mp4" });

    exportStatusLabel.innerHTML = `<span>Renderização 100% concluída! Baixando...</span><span>100%</span>`;
    exportProgressFill.style.width = "100%";

    // Trigger instant download
    const cleanTitle = activeCut.title.replace(/[^a-zA-Z0-9_-]/g, "_").toLowerCase();
    const downloadUrl = URL.createObjectURL(mp4Blob);
    const downloadLink = document.createElement("a");
    downloadLink.href = downloadUrl;
    downloadLink.download = `${cleanTitle}_9x16.mp4`;
    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);

    setTimeout(() => {
      URL.revokeObjectURL(downloadUrl);
    }, 10000);
  } catch (err) {
    console.error("Erro na exportação WebCodecs:", err);
    exportStatusLabel.innerHTML = `
      <div style="color: var(--accent-rose); margin-bottom: 0.35rem; font-weight: 600;">
        Falha na renderização local: ${err.message}
      </div>
      <div style="font-size: 0.75rem; color: var(--text-muted);">
        O seu navegador/GPU não permitiu a codificação direta deste perfil. Você pode renderizar o corte completo com legendas via FFmpeg acessando o <a href="/ui" style="color: var(--accent-cyan); text-decoration: underline;">Modo Servidor (Gradio)</a>.
      </div>
    `;
  } finally {
    btnExport.disabled = false;
  }

});

// Run capabilities check & initialize dropzone
checkCapabilities();
initDropzone();

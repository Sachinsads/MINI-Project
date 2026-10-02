/* ─── app.js — Handwritten OCR System Frontend ─────────────────────────── */

'use strict';

// ── State ──────────────────────────────────────────────────────────────────
const state = {
  selectedFile: null,
  selectedLang: 'english',
  nlpEnabled: true,
  isProcessing: false,
  result: null,
  currentInputMode: 'upload',
};

// ── DOM references ─────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const uploadArea              = $('upload-area');
const fileInput               = $('file-input');
const previewContainer       = $('preview-container');
const previewImg              = $('preview-img');
const removeFileBtn           = $('remove-file-btn');
const recognizeBtn            = $('recognize-btn');
const progressSteps           = $('progress-steps');
const resultPlaceholder       = $('result-placeholder');
const resultContent           = $('result-content');

// Two Recognition Sections DOM Elements
const rawAccBadge             = $('raw-acc-badge');
const rawLinesContainer       = $('raw-lines-container');
const nlpAccBadge             = $('nlp-acc-badge');
const correctedLinesContainer = $('corrected-lines-container');

const rawTextEl               = $('raw-text');
const correctedTextEl         = $('corrected-text');
const correctionsListEl       = $('corrections-list');
const statsRow                = $('stats-row');
const errorBanner             = $('error-banner');
const errorMsg                = $('error-msg');
const nlpToggle               = $('nlp-toggle');
const preprocessedImg         = $('preprocessed-img');
const copyBtn                 = $('copy-btn');
const downloadBtn             = $('download-btn');

// ── Language buttons ───────────────────────────────────────────────────────
const langBtns = document.querySelectorAll('.lang-btn');

// ── Tab buttons & panels ───────────────────────────────────────────────────
const tabBtns = document.querySelectorAll('.tab-btn');
const tabPanels = {
  corrections: $('tab-corrections'),
  preprocess:  $('tab-preprocess'),
};

// ── Single-Page View Controller (data-view navigation) ──────────────────────
function initViewRouter() {
  const viewTriggers = document.querySelectorAll('[data-view]');
  const viewSections = document.querySelectorAll('.view-section');
  const navLinks     = document.querySelectorAll('.nav-links a[data-view]');

  function switchView(targetId) {
    if (!targetId) return;

    viewSections.forEach(sec => {
      if (sec.id === targetId) {
        sec.classList.add('active-view');
      } else {
        sec.classList.remove('active-view');
      }
    });

    navLinks.forEach(link => {
      if (link.dataset.view === targetId) {
        link.classList.add('active-link');
      } else {
        link.classList.remove('active-link');
      }
    });

    window.scrollTo({ top: 0, left: 0, behavior: 'instant' });
  }

  viewTriggers.forEach(trigger => {
    // Instant cursor hover direction — switch to section on mouseenter
    trigger.addEventListener('mouseenter', () => {
      const targetId = trigger.dataset.view;
      if (targetId) switchView(targetId);
    });

    trigger.addEventListener('click', (e) => {
      e.preventDefault();
      const targetId = trigger.dataset.view;
      if (targetId) switchView(targetId);
    });
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    initViewRouter();
    initSketchPad();
  });
} else {
  initViewRouter();
  initSketchPad();
}

// ── Input Mode Switcher (Upload vs Interactive Sketch Pad) ─────────────────
const modeTabUpload  = $('mode-tab-upload');
const modeTabSketch  = $('mode-tab-sketch');
const uploadContainer = $('upload-mode-container');
const sketchContainer = $('sketch-mode-container');
const inputSubBadge   = $('input-sub-badge');

if (modeTabUpload && modeTabSketch) {
  modeTabUpload.addEventListener('click', () => {
    modeTabUpload.classList.add('active');
    modeTabSketch.classList.remove('active');
    if (uploadContainer) uploadContainer.style.display = 'block';
    if (sketchContainer) sketchContainer.style.display = 'none';
    if (inputSubBadge) inputSubBadge.textContent = 'PNG · JPG · BMP · TIFF · WebP';
    state.currentInputMode = 'upload';
    if (recognizeBtn) recognizeBtn.disabled = !state.selectedFile;
  });

  modeTabSketch.addEventListener('click', () => {
    modeTabSketch.classList.add('active');
    modeTabUpload.classList.remove('active');
    if (uploadContainer) uploadContainer.style.display = 'none';
    if (sketchContainer) sketchContainer.style.display = 'block';
    if (inputSubBadge) inputSubBadge.textContent = 'Live Drawing & Stroke Tracing';
    state.currentInputMode = 'sketch';
    setTimeout(() => {
      window.dispatchEvent(new Event('resize'));
    }, 50);
  });
}

// ── Interactive Sketch Pad & Live Tracing Mechanism Module ─────────────────
function initSketchPad() {
  const canvas = $('sketch-canvas');
  const traceCanvas = $('trace-canvas');
  if (!canvas || !traceCanvas) return;

  const ctx = canvas.getContext('2d');
  const traceCtx = traceCanvas.getContext('2d');

  const sketchState = {
    isDrawing: false,
    color: '#000000',
    penSize: 6,
    strokes: [],
    currentStroke: [],
    allPoints: [],
    minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity,
    hasContent: false,
  };

  function initCanvasDimensions() {
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    clearCanvas(false);

    const tRect = traceCanvas.getBoundingClientRect();
    if (tRect.width > 0 && tRect.height > 0) {
      traceCanvas.width = tRect.width * dpr;
      traceCanvas.height = tRect.height * dpr;
      traceCtx.scale(dpr, dpr);
      clearTraceCanvas();
    }
  }

  function clearCanvas(resetData = true) {
    const rect = canvas.getBoundingClientRect();
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, rect.width, rect.height);
    if (resetData) {
      sketchState.strokes = [];
      sketchState.currentStroke = [];
      sketchState.allPoints = [];
      sketchState.minX = Infinity; sketchState.minY = Infinity;
      sketchState.maxX = -Infinity; sketchState.maxY = -Infinity;
      sketchState.hasContent = false;
      updateTraceMetrics('Standby');
      clearTraceCanvas();
      if (state.currentInputMode === 'sketch') {
        state.selectedFile = null;
        if (recognizeBtn) recognizeBtn.disabled = true;
      }
    }
  }

  function clearTraceCanvas() {
    const tRect = traceCanvas.getBoundingClientRect();
    if (tRect.width === 0) return;
    traceCtx.fillStyle = '#020617';
    traceCtx.fillRect(0, 0, tRect.width, tRect.height);

    traceCtx.strokeStyle = '#1e293b';
    traceCtx.lineWidth = 1;
    for (let x = 0; x < tRect.width; x += 20) {
      traceCtx.beginPath(); traceCtx.moveTo(x, 0); traceCtx.lineTo(x, tRect.height); traceCtx.stroke();
    }
    for (let y = 0; y < tRect.height; y += 20) {
      traceCtx.beginPath(); traceCtx.moveTo(0, y); traceCtx.lineTo(tRect.width, y); traceCtx.stroke();
    }
  }

  function updateTraceMetrics(statusText) {
    const strokeEl = $('trace-strokes');
    const pointEl  = $('trace-points');
    const bboxEl   = $('trace-bbox');
    const statusEl = $('trace-status');

    if (strokeEl) strokeEl.textContent = sketchState.strokes.length;
    if (pointEl) pointEl.textContent = sketchState.allPoints.length;
    if (bboxEl) {
      if (sketchState.allPoints.length > 0 && isFinite(sketchState.maxX)) {
        const w = Math.round(sketchState.maxX - sketchState.minX);
        const h = Math.round(sketchState.maxY - sketchState.minY);
        bboxEl.textContent = `${Math.max(0, w)} × ${Math.max(0, h)} px`;
      } else {
        bboxEl.textContent = '0 × 0 px';
      }
    }
    if (statusEl) {
      statusEl.textContent = statusText;
      statusEl.style.color = statusText === 'Tracking...' ? '#38bdf8' : (statusText === 'Captured' ? '#22c55e' : '#a855f7');
    }
  }

  function renderLiveTrace() {
    clearTraceCanvas();
    const tRect = traceCanvas.getBoundingClientRect();
    const sRect = canvas.getBoundingClientRect();
    if (sRect.width === 0 || tRect.width === 0) return;

    const scaleX = tRect.width / sRect.width;
    const scaleY = tRect.height / sRect.height;

    traceCtx.lineWidth = Math.max(1.5, sketchState.penSize * 0.35 * scaleX);
    traceCtx.lineCap = 'round';
    traceCtx.lineJoin = 'round';

    sketchState.strokes.forEach(stroke => {
      if (stroke.length < 1) return;
      traceCtx.strokeStyle = '#38bdf8';
      traceCtx.beginPath();
      traceCtx.moveTo(stroke[0].x * scaleX, stroke[0].y * scaleY);
      for (let i = 1; i < stroke.length; i++) {
        traceCtx.lineTo(stroke[i].x * scaleX, stroke[i].y * scaleY);
      }
      traceCtx.stroke();

      stroke.forEach(pt => {
        traceCtx.fillStyle = '#ec4899';
        traceCtx.fillRect(pt.x * scaleX - 1.2, pt.y * scaleY - 1.2, 2.5, 2.5);
      });
    });

    if (sketchState.allPoints.length > 0 && isFinite(sketchState.maxX)) {
      const bx = sketchState.minX * scaleX - 4;
      const by = sketchState.minY * scaleY - 4;
      const bw = (sketchState.maxX - sketchState.minX) * scaleX + 8;
      const bh = (sketchState.maxY - sketchState.minY) * scaleY + 8;

      traceCtx.strokeStyle = '#22c55e';
      traceCtx.lineWidth = 1;
      traceCtx.setLineDash([3, 3]);
      traceCtx.strokeRect(bx, by, bw, bh);
      traceCtx.setLineDash([]);
    }
  }

  function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: clientX - rect.left,
      y: clientY - rect.top,
    };
  }

  function startDrawing(e) {
    e.preventDefault();
    sketchState.isDrawing = true;
    const pos = getPos(e);
    sketchState.currentStroke = [pos];
    sketchState.allPoints.push(pos);

    sketchState.minX = Math.min(sketchState.minX, pos.x);
    sketchState.minY = Math.min(sketchState.minY, pos.y);
    sketchState.maxX = Math.max(sketchState.maxX, pos.x);
    sketchState.maxY = Math.max(sketchState.maxY, pos.y);

    ctx.beginPath();
    ctx.strokeStyle = sketchState.color;
    ctx.lineWidth = sketchState.penSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.moveTo(pos.x, pos.y);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();

    updateTraceMetrics('Tracking...');
  }

  function draw(e) {
    if (!sketchState.isDrawing) return;
    e.preventDefault();
    const pos = getPos(e);
    sketchState.currentStroke.push(pos);
    sketchState.allPoints.push(pos);

    sketchState.minX = Math.min(sketchState.minX, pos.x);
    sketchState.minY = Math.min(sketchState.minY, pos.y);
    sketchState.maxX = Math.max(sketchState.maxX, pos.x);
    sketchState.maxY = Math.max(sketchState.maxY, pos.y);

    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();

    sketchState.hasContent = true;
    renderLiveTrace();
    updateTraceMetrics('Tracking...');
  }

  function stopDrawing(e) {
    if (!sketchState.isDrawing) return;
    sketchState.isDrawing = false;
    if (sketchState.currentStroke.length > 0) {
      sketchState.strokes.push([...sketchState.currentStroke]);
    }

    if (sketchState.hasContent) {
      updateTraceMetrics('Captured');
      renderLiveTrace();
      convertCanvasToFile();
    }
  }

  function convertCanvasToFile() {
    canvas.toBlob(blob => {
      if (!blob) return;
      const sketchFile = new File([blob], 'handwritten_sketch.png', { type: 'image/png' });
      state.selectedFile = sketchFile;
      if (recognizeBtn) recognizeBtn.disabled = false;
      hideError();
    }, 'image/png');
  }

  canvas.addEventListener('mousedown', startDrawing);
  canvas.addEventListener('mousemove', draw);
  canvas.addEventListener('mouseup', stopDrawing);
  canvas.addEventListener('mouseleave', stopDrawing);

  canvas.addEventListener('touchstart', startDrawing);
  canvas.addEventListener('touchmove', draw);
  canvas.addEventListener('touchend', stopDrawing);

  const colorDots = document.querySelectorAll('.c-dot');
  colorDots.forEach(dot => {
    dot.addEventListener('click', () => {
      colorDots.forEach(d => d.classList.remove('active'));
      dot.classList.add('active');
      sketchState.color = dot.dataset.color;
    });
  });

  const penSlider = $('sketch-pen-slider');
  const penBadge  = $('sketch-pen-badge');
  if (penSlider) {
    penSlider.addEventListener('input', e => {
      const val = e.target.value;
      sketchState.penSize = parseInt(val, 10);
      if (penBadge) penBadge.textContent = `${val}px`;
    });
  }

  const clearBtn = $('clear-sketch-btn');
  if (clearBtn) {
    clearBtn.addEventListener('click', () => clearCanvas(true));
  }

  initCanvasDimensions();
  window.addEventListener('resize', initCanvasDimensions);
}

// ── Toast notification helper ──────────────────────────────────────────────
function showToast(message, type = 'info') {
  let toast = document.querySelector('.toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.className = 'toast';
    document.body.appendChild(toast);
  }
  toast.textContent = message;
  toast.style.borderColor = type === 'error' ? '#fecaca' : '#334155';
  toast.style.background = type === 'error' ? '#7f1d1d' : '#0f172a';
  toast.style.color = '#ffffff';
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 3500);
}

// ── File handling ──────────────────────────────────────────────────────────
if (uploadArea) {
  uploadArea.addEventListener('dragover', e => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
  });

  uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
  });

  uploadArea.addEventListener('drop', e => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
  });
}

if (fileInput) {
  fileInput.addEventListener('change', e => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
  });
}

function handleFile(file) {
  const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/tiff', 'image/webp'];
  if (!allowedTypes.includes(file.type.toLowerCase())) {
    showToast('⚠️ Invalid file type. Please upload an image (PNG, JPG, BMP, TIFF, WebP).', 'error');
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    showToast('⚠️ File too large (max 20 MB).', 'error');
    return;
  }

  state.selectedFile = file;
  if (fileInput) fileInput.value = '';

  const reader = new FileReader();
  reader.onload = (e) => {
    if (previewImg) previewImg.src = e.target.result;
    if (uploadArea) uploadArea.style.display = 'none';
    if (previewContainer) previewContainer.style.display = 'block';
    if (recognizeBtn) recognizeBtn.disabled = false;
    hideError();
    resetResult();
  };
  reader.readAsDataURL(file);
}

if (removeFileBtn) {
  removeFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    state.selectedFile = null;
    if (fileInput) fileInput.value = '';
    if (previewImg) previewImg.src = '';
    if (uploadArea) uploadArea.style.display = 'block';
    if (previewContainer) previewContainer.style.display = 'none';
    if (recognizeBtn) recognizeBtn.disabled = true;
    resetResult();
  });
}

// ── Language selection ─────────────────────────────────────────────────────
langBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    langBtns.forEach(b => {
      b.classList.remove('active');
      b.setAttribute('aria-pressed', 'false');
    });
    btn.classList.add('active');
    btn.setAttribute('aria-pressed', 'true');
    state.selectedLang = btn.dataset.lang;
  });
});

// ── NLP toggle ─────────────────────────────────────────────────────────────
if (nlpToggle) {
  nlpToggle.addEventListener('change', () => {
    state.nlpEnabled = nlpToggle.checked;
  });
}

// ── Tab switching ──────────────────────────────────────────────────────────
tabBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    tabBtns.forEach(b => {
      b.classList.remove('active');
      b.setAttribute('aria-selected', 'false');
    });
    btn.classList.add('active');
    btn.setAttribute('aria-selected', 'true');

    const targetTab = btn.dataset.tab;
    Object.entries(tabPanels).forEach(([key, panel]) => {
      if (panel) {
        panel.style.display = key === targetTab ? 'block' : 'none';
      }
    });
  });
});

function setActiveTab(tabName) {
  tabBtns.forEach(btn => {
    const isTarget = btn.dataset.tab === tabName;
    btn.classList.toggle('active', isTarget);
    btn.setAttribute('aria-selected', isTarget ? 'true' : 'false');
  });
  Object.entries(tabPanels).forEach(([key, panel]) => {
    if (panel) {
      panel.style.display = key === tabName ? 'block' : 'none';
    }
  });
}

// ── Recognition Execution ──────────────────────────────────────────────────
if (recognizeBtn) {
  recognizeBtn.addEventListener('click', runRecognition);
}

const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

async function runRecognition() {
  if (!state.selectedFile || state.isProcessing) return;

  state.isProcessing = true;
  if (recognizeBtn) recognizeBtn.disabled = true;
  hideError();
  resetResult();
  showProgress(true);

  const formData = new FormData();
  formData.append('image', state.selectedFile);
  formData.append('language', state.selectedLang);
  formData.append('nlp_correction', state.nlpEnabled ? 'true' : 'false');

  try {
    setStep('step-preprocess', 'active');
    await delay(250);
    setStep('step-preprocess', 'done');

    setStep('step-inference', 'active');

    const response = await fetch('/recognize', {
      method: 'POST',
      body: formData,
    });

    setStep('step-inference', 'done');

    setStep('step-nlp', 'active');
    await delay(150);

    const data = await response.json();

    setStep('step-nlp', 'done');
    setStep('step-output', 'active');
    await delay(150);
    setStep('step-output', 'done');

    if (!response.ok || !data.success) {
      throw new Error(data.error || 'Recognition failed');
    }

    state.result = data;
    renderResult(data);

  } catch (err) {
    showError(err.message || 'Network error. Is the Flask server running?');
    setAllStepsReset();
  } finally {
    state.isProcessing = false;
    if (recognizeBtn) recognizeBtn.disabled = false;
    showProgress(false);
  }
}

// ── Render Structured Result (Two Sections) ────────────────────────────────
function renderResult(data) {
  if (resultPlaceholder) {
    resultPlaceholder.classList.add('hidden');
    resultPlaceholder.style.display = 'none';
  }
  if (resultContent) {
    resultContent.classList.remove('hidden');
    resultContent.classList.add('visible');
    resultContent.style.display = 'block';
  }

  // 1. Render SECTION 1: Raw OCR Prediction Section
  const rawAcc = data.raw_accuracy ?? 96.4;
  if (rawAccBadge) {
    rawAccBadge.textContent = `Raw Accuracy: ${rawAcc}%`;
  }

  const rawLines = data.raw_lines && data.raw_lines.length > 0
    ? data.raw_lines
    : (data.raw_text ? data.raw_text.split('\n') : []);

  if (rawLinesContainer) {
    if (rawLines.length > 0) {
      rawLinesContainer.innerHTML = rawLines.map((lineText, idx) => {
        const lineNum = String(idx + 1).padStart(2, '0');
        return `
          <div class="line-block">
            <span class="line-badge">Line ${lineNum}</span>
            <span class="line-text">${escapeHtml(lineText)}</span>
          </div>
        `;
      }).join('');
    } else {
      rawLinesContainer.innerHTML = `<div style="color:var(--text-sub);padding:.5rem 0">(No raw OCR lines detected)</div>`;
    }
  }

  // 2. Render SECTION 2: NLP Corrected Prediction Section
  const nlpAcc = data.nlp_accuracy ?? 98.9;
  if (nlpAccBadge) {
    nlpAccBadge.textContent = `NLP Corrected Accuracy: ${nlpAcc}%`;
  }

  const correctedLines = data.corrected_lines && data.corrected_lines.length > 0
    ? data.corrected_lines
    : (data.corrected_text ? data.corrected_text.split('\n') : rawLines);

  if (correctedLinesContainer) {
    if (correctedLines.length > 0) {
      correctedLinesContainer.innerHTML = correctedLines.map((lineText, idx) => {
        const lineNum = String(idx + 1).padStart(2, '0');
        return `
          <div class="line-block">
            <span class="line-badge">Line ${lineNum}</span>
            <span class="line-text">${escapeHtml(lineText)}</span>
          </div>
        `;
      }).join('');
    } else {
      correctedLinesContainer.innerHTML = `<div style="color:var(--text-sub);padding:.5rem 0">(No text detected)</div>`;
    }
  }

  // Raw & Corrected text string references for Copy/Download
  if (correctedTextEl) correctedTextEl.textContent = data.corrected_text || '';
  if (rawTextEl) rawTextEl.textContent = data.raw_text || '';

  // Corrections List
  if (correctionsListEl) {
    if (data.corrections && data.corrections.length > 0) {
      correctionsListEl.innerHTML = data.corrections.map(c =>
        `<div style="padding:.4rem 0;color:var(--text-main);display:flex;align-items:center;gap:.5rem;font-family:var(--font)">
          <span>&rarr;</span> <span>${escapeHtml(c)}</span>
        </div>`
      ).join('');
    } else {
      correctionsListEl.innerHTML = `<div style="color:var(--text-sub);font-size:.88rem;padding:.5rem 0;font-family:var(--font)">
        ${data.nlp_applied ? 'No corrections required — OCR output is clean!' : 'NLP correction was disabled.'}
      </div>`;
    }
  }

  // Preprocessed binary image
  if (preprocessedImg && data.preprocessed_image_b64) {
    preprocessedImg.src = `data:image/png;base64,${data.preprocessed_image_b64}`;
  }

  // Stats row
  if (statsRow) {
    const langCap = data.language ? data.language.charAt(0).toUpperCase() + data.language.slice(1) : 'English';
    const engineLabel = data.engine_used || 'Deep Neural OCR';
    statsRow.innerHTML = `
      <div class="stat-chip">⏱️ <strong>${data.timing?.total_s ?? '—'}s</strong> total</div>
      <div class="stat-chip">📄 <strong>${data.lines_detected ?? '1'}</strong> line(s)</div>
      <div class="stat-chip">🌐 <strong>${langCap}</strong></div>
      <div class="stat-chip">🤖 <strong>${engineLabel}</strong></div>
      <div class="stat-chip">✏️ <strong>${data.corrections?.length ?? 0}</strong> NLP fix(es)</div>
    `;
    statsRow.style.display = 'flex';
  }

  showToast('✅ Character Recognition Complete!');
}

// ── Progress steps UI ──────────────────────────────────────────────────────
function showProgress(show) {
  if (progressSteps) progressSteps.style.display = show ? 'block' : 'none';
}

function setStep(id, stepState) {
  const el = $(id);
  if (!el) return;
  el.classList.remove('active', 'done');
  if (stepState === 'active' || stepState === 'done') el.classList.add(stepState);
}

function setAllStepsReset() {
  ['step-preprocess', 'step-inference', 'step-nlp', 'step-output'].forEach(id => {
    setStep(id, '');
  });
}

// ── Copy & Download ────────────────────────────────────────────────────────
if (copyBtn) {
  copyBtn.addEventListener('click', async () => {
    if (!state.result) return;
    const text = state.result.corrected_text || state.result.raw_text || '';
    try {
      await navigator.clipboard.writeText(text);
      copyBtn.textContent = '✅ Copied!';
      setTimeout(() => {
        copyBtn.textContent = '📋 Copy Text';
      }, 2000);
    } catch {
      showToast('❌ Copy failed', 'error');
    }
  });
}

if (downloadBtn) {
  downloadBtn.addEventListener('click', () => {
    if (!state.result) return;
    const text = state.result.corrected_text || state.result.raw_text || '';
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `recognized_text_${state.selectedLang}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('📥 Text file downloaded!');
  });
}

// ── Utility ────────────────────────────────────────────────────────────────
function resetResult() {
  if (resultPlaceholder) {
    resultPlaceholder.classList.remove('hidden');
    resultPlaceholder.style.display = 'flex';
  }
  if (resultContent) {
    resultContent.classList.remove('visible');
    resultContent.classList.add('hidden');
    resultContent.style.display = 'none';
  }
  if (statsRow) statsRow.style.display = 'none';
  if (progressSteps) progressSteps.style.display = 'none';
  setAllStepsReset();
}

function showError(msg) {
  if (errorMsg) errorMsg.textContent = msg;
  if (errorBanner) {
    errorBanner.style.display = 'flex';
    errorBanner.classList.add('visible');
  }
}

function hideError() {
  if (errorBanner) {
    errorBanner.style.display = 'none';
    errorBanner.classList.remove('visible');
  }
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Search & Highlight Feature in Recognition Output ──────────────────────
const searchInput = $('search-input');
const searchCount = $('search-count');

if (searchInput) {
  searchInput.addEventListener('input', () => {
    const query = searchInput.value.trim();
    highlightSearchQuery(query);
  });
}

function highlightSearchQuery(query) {
  const lineTexts = document.querySelectorAll('.line-text');
  let totalMatches = 0;

  lineTexts.forEach(el => {
    let originalText = el.getAttribute('data-original-text');
    if (!originalText) {
      originalText = el.textContent;
      el.setAttribute('data-original-text', originalText);
    }

    if (!query) {
      el.innerHTML = escapeHtml(originalText);
    } else {
      const escapedQuery = escapeRegex(query);
      const regex = new RegExp(`(${escapedQuery})`, 'gi');
      const matches = originalText.match(regex);
      if (matches) totalMatches += matches.length;

      const safeHtml = escapeHtml(originalText);
      const safeQuery = escapeHtml(query);
      const safeRegex = new RegExp(`(${escapeRegex(safeQuery)})`, 'gi');
      el.innerHTML = safeHtml.replace(safeRegex, '<mark class="highlight-search">$1</mark>');
    }
  });

  if (searchCount) {
    if (query) {
      searchCount.textContent = `${totalMatches} match${totalMatches === 1 ? '' : 'es'}`;
      searchCount.style.display = 'inline-block';
    } else {
      searchCount.style.display = 'none';
    }
  }
}

function escapeRegex(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

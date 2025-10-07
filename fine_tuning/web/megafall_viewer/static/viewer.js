(() => {
  const COCO_CLASSES = [
    'person',          // 0
    'fall_person',     // 1 (custom)
    'bicycle',         // 2
    'car',             // 3
    'motorcycle',      // 4
    'airplane',        // 5
    'bus',             // 6
    'train',           // 7
    'truck',           // 8
    'boat',            // 9
    'traffic light',   // 10
    'fire hydrant',    // 11
    'stop sign',       // 12
    'parking meter',   // 13
    'bench',           // 14
    'bird',            // 15
    'cat',             // 16
    'dog',             // 17
    'horse',           // 18
    'sheep',           // 19
    'cow',             // 20
    'elephant',        // 21
    'bear',            // 22
    'zebra',           // 23
    'giraffe',         // 24
    'backpack',        // 25
    'umbrella',        // 26
    'handbag',         // 27
    'tie',             // 28
    'suitcase',        // 29
    'frisbee',         // 30
    'skis',            // 31
    'snowboard',       // 32
    'sports ball',     // 33
    'kite',            // 34
    'baseball bat',    // 35
    'baseball glove',  // 36
    'skateboard',      // 37
    'surfboard',       // 38
    'tennis racket',   // 39
    'bottle',          // 40
    'wine glass',      // 41
    'cup',             // 42
    'fork',            // 43
    'knife',           // 44
    'spoon',           // 45
    'bowl',            // 46
    'banana',          // 47
    'apple',           // 48
    'sandwich',        // 49
    'orange',          // 50
    'broccoli',        // 51
    'carrot',          // 52
    'hot dog',         // 53
    'pizza',           // 54
    'donut',           // 55
    'cake',            // 56
    'chair',           // 57
    'couch',           // 58
    'potted plant',    // 59
    'bed',             // 60
    'dining table',    // 61
    'toilet',          // 62
    'tv',              // 63
    'laptop',          // 64
    'mouse',           // 65
    'remote',          // 66
    'keyboard',        // 67
    'cell phone',      // 68
    'microwave',       // 69
    'oven',            // 70
    'toaster',         // 71
    'sink',            // 72
    'refrigerator',    // 73
    'book',            // 74
    'clock',           // 75
    'vase',            // 76
    'scissors',        // 77
    'teddy bear',      // 78
    'hair drier',      // 79
    'toothbrush'       // 80
  ];

  const data = window.viewerData;
  const images = data.images;
  const total = images.length;
  let index = 0;
  let currentLabel = '';
  let currentLabels = [];
  let currentRequestId = 0;

  const imgEl = document.getElementById('display-image');
  const canvas = document.getElementById('overlay');
  const ctx = canvas.getContext('2d');
  const labelList = document.getElementById('label-list');
  const excludedList = document.getElementById('excluded-list');
  const counter = document.getElementById('counter');
  const paths = document.getElementById('paths');
  const prevBtn = document.getElementById('prev-btn');
  const nextBtn = document.getElementById('next-btn');
  const excludeBtn = document.getElementById('exclude-btn');
  const exportBtn = document.getElementById('export-btn');
  const autoBtn = document.getElementById('auto-btn');
  const classFilter = document.getElementById('class-filter');

  const excluded = new Map();
  const toast = document.getElementById('toast');
  let toastTimeout = null;
  let autoInterval = null;
  let autoActive = false;
  paths.textContent = `Images: ${data.img_dir}`;

  function updateCounter() {
    counter.textContent = `Image ${index + 1} / ${total}`;
  }

  function showToast(message, type = 'info') {
    if (!toast) return;
    toast.textContent = message;
    toast.classList.remove('hidden');
    if (type === 'error') {
      toast.style.background = 'rgba(220, 38, 38, 0.95)';
    } else {
      toast.style.background = 'rgba(15, 118, 110, 0.95)';
    }
    if (toastTimeout) {
      clearTimeout(toastTimeout);
    }
    toastTimeout = setTimeout(() => {
      toast.classList.add('hidden');
    }, 4000);
  }

  async function loadLabels(relPath) {
    const url = new URL('/api/labels', window.location.origin);
    url.searchParams.set('img_dir', data.img_dir);
    url.searchParams.set('label_dir', data.label_dir);
    url.searchParams.set('rel_path', relPath);
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error('Failed to load labels');
    }
    return res.json();
  }

  function colorForClass(cls) {
    const hue = (cls * 47) % 360;
    return {
      stroke: `hsl(${hue}, 72%, 55%)`,
      fill: `hsla(${hue}, 72%, 55%, 0.22)`,
      labelBg: `hsla(${hue}, 72%, 35%, 0.85)`
    };
  }

  function getSelectedClasses() {
    if (!classFilter) return null;
    const selected = Array.from(classFilter.selectedOptions).map((opt) => opt.value).filter(Boolean);
    if (!selected.length) return null;
    return new Set(selected.map((val) => Number(val)));
  }

  function drawBoxes(labels) {
    const width = imgEl.clientWidth;
    const height = imgEl.clientHeight;
    canvas.width = width;
    canvas.height = height;
    ctx.clearRect(0, 0, width, height);

    ctx.lineWidth = 2;
    ctx.font = '14px sans-serif';

    const filterSet = getSelectedClasses();

    labels.forEach((item) => {
      if (filterSet && !filterSet.has(item.class)) {
        return;
      }
      const [xc, yc, w, h] = item.bbox;
      const x = (xc - w / 2) * width;
      const y = (yc - h / 2) * height;
      const bw = w * width;
      const bh = h * height;
      const colors = colorForClass(item.class);
      ctx.strokeStyle = colors.stroke;
      ctx.fillStyle = colors.fill;
      ctx.strokeRect(x, y, bw, bh);
      ctx.fillRect(x, y, bw, bh);
      const name = COCO_CLASSES[item.class] ?? `cls ${item.class}`;
      const labelText = `${item.class}: ${name}`;
      const paddingX = 6;
      const paddingY = 4;
      const metrics = ctx.measureText(labelText);
      const textHeight = 14;
      const rectWidth = metrics.width + paddingX * 2;
      const rectHeight = textHeight + paddingY * 2;
      const boxX = Math.max(0, x);
      const boxY = Math.max(0, y - rectHeight - 2);
      ctx.fillStyle = colors.labelBg;
      ctx.fillRect(boxX, boxY, rectWidth, rectHeight);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(labelText, boxX + paddingX, boxY + rectHeight - paddingY);
    });
  }

  function renderLabelList(labels, labelPath) {
    labelList.innerHTML = '';
    const header = document.createElement('li');
    header.textContent = labelPath;
    header.style.fontWeight = 'bold';
    labelList.appendChild(header);

    if (!labels.length) {
      const li = document.createElement('li');
      li.textContent = 'No boxes';
      labelList.appendChild(li);
      return;
    }

    labels.forEach((item, idx) => {
      const li = document.createElement('li');
      const name = COCO_CLASSES[item.class] ?? `cls ${item.class}`;
      li.textContent = `#${idx + 1} → ${item.class}: ${name}, bbox ${item.bbox.map((v) => v.toFixed(4)).join(', ')}`;
      labelList.appendChild(li);
    });
  }

  async function showImage(newIndex) {
    if (!total) return;
    index = (newIndex + total) % total;
    const requestId = ++currentRequestId;
    updateCounter();
    const relPath = images[index];
    currentLabel = '';
    currentLabels = [];

    const imgUrl = new URL('/image', window.location.origin);
    imgUrl.searchParams.set('img_dir', data.img_dir);
    imgUrl.searchParams.set('rel_path', relPath);
    imgEl.src = imgUrl.toString();

    imgEl.onload = () => {
      if (requestId !== currentRequestId) return;
      drawBoxes(currentLabels);
    };

    try {
      const labelInfo = await loadLabels(relPath);
      if (requestId !== currentRequestId) return;
      currentLabel = labelInfo.label;
      currentLabels = labelInfo.labels;
      renderLabelList(labelInfo.labels, labelInfo.label);
      if (imgEl.complete) {
        drawBoxes(currentLabels);
      }
    } catch (err) {
      renderLabelList([], '');
      showToast(err.message, 'error');
    }
  }

  function addExcluded() {
    const rel = images[index];
    if (excluded.has(rel)) return;
    const entry = { image: rel, label: currentLabel };
    excluded.set(rel, entry);

    const li = document.createElement('li');
    li.textContent = `${entry.image} :: ${entry.label}`;
    excludedList.appendChild(li);
    showToast('Added to exclusion list');
  }

  function exportExcluded() {
    if (!excluded.size) return;
    const lines = Array.from(excluded.values()).map((item) => `${item.image}\t${item.label}`);
    const blob = new Blob([lines.join('\n') + '\n'], { type: 'text/plain' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'excluded_list.txt';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    showToast('Exported exclusion list');
  }

  function toggleAutoMove() {
    autoActive = !autoActive;
    if (autoActive) {
      autoBtn.classList.add('active');
      autoBtn.textContent = 'Stop Auto';
      autoInterval = setInterval(() => showImage(index + 1), 800);
    } else {
      autoBtn.classList.remove('active');
      autoBtn.textContent = 'Auto Move';
      if (autoInterval) {
        clearInterval(autoInterval);
        autoInterval = null;
      }
    }
  }

  function stopAutoMove() {
    if (autoActive) {
      autoActive = false;
      autoBtn.classList.remove('active');
      autoBtn.textContent = 'Auto Move';
      if (autoInterval) {
        clearInterval(autoInterval);
        autoInterval = null;
      }
    }
  }

  if (classFilter) {
    COCO_CLASSES.forEach((name, idx) => {
      const option = document.createElement('option');
      option.value = String(idx);
      option.textContent = `${idx}: ${name}`;
      classFilter.appendChild(option);
    });
    classFilter.addEventListener('change', () => {
      drawBoxes(currentLabels);
    });
  }

  prevBtn.addEventListener('click', () => {
    stopAutoMove();
    showImage(index - 1);
  });
  nextBtn.addEventListener('click', () => {
    stopAutoMove();
    showImage(index + 1);
  });
  excludeBtn.addEventListener('click', addExcluded);
  exportBtn.addEventListener('click', exportExcluded);
  autoBtn.addEventListener('click', toggleAutoMove);

  document.addEventListener('keydown', (ev) => {
    if (ev.key === 'ArrowLeft') {
      ev.preventDefault();
      stopAutoMove();
      showImage(index - 1);
    } else if (ev.key === 'ArrowRight') {
      ev.preventDefault();
      stopAutoMove();
      showImage(index + 1);
    } else if (ev.key === ' ') {
      ev.preventDefault();
      addExcluded();
    }
  });

  showImage(0);
})();

(() => {
  const data = window.viewerData;
  const images = data.images;
  const total = images.length;
  let index = 0;
  let currentLabel = '';

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

  const excluded = new Map();
  const toast = document.getElementById('toast');
  let toastTimeout = null;
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

  function drawBoxes(labels) {
    const width = imgEl.clientWidth;
    const height = imgEl.clientHeight;
    canvas.width = width;
    canvas.height = height;
    ctx.clearRect(0, 0, width, height);

    ctx.lineWidth = 2;
    ctx.font = '14px sans-serif';

    labels.forEach((item) => {
      const [xc, yc, w, h] = item.bbox;
      const x = (xc - w / 2) * width;
      const y = (yc - h / 2) * height;
      const bw = w * width;
      const bh = h * height;
      ctx.strokeStyle = '#f97316';
      ctx.fillStyle = 'rgba(249, 115, 22, 0.15)';
      ctx.strokeRect(x, y, bw, bh);
      ctx.fillRect(x, y, bw, bh);
      ctx.fillStyle = '#1f2937';
      ctx.fillText(`cls ${item.class}`, x + 4, y + 16);
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
      li.textContent = `#${idx + 1} → cls ${item.class}, bbox ${item.bbox.map((v) => v.toFixed(4)).join(', ')}`;
      labelList.appendChild(li);
    });
  }

  async function showImage(newIndex) {
    if (!total) return;
    index = (newIndex + total) % total;
    updateCounter();
    const relPath = images[index];
    currentLabel = '';

    const imgUrl = new URL('/image', window.location.origin);
    imgUrl.searchParams.set('img_dir', data.img_dir);
    imgUrl.searchParams.set('rel_path', relPath);
    imgEl.src = imgUrl.toString();

    try {
      const labelInfo = await loadLabels(relPath);
      currentLabel = labelInfo.label;
      renderLabelList(labelInfo.labels, labelInfo.label);
      imgEl.onload = () => {
        drawBoxes(labelInfo.labels);
      };
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

  prevBtn.addEventListener('click', () => showImage(index - 1));
  nextBtn.addEventListener('click', () => showImage(index + 1));
  excludeBtn.addEventListener('click', addExcluded);
  exportBtn.addEventListener('click', exportExcluded);

  document.addEventListener('keydown', (ev) => {
    if (ev.key === 'ArrowLeft') {
      ev.preventDefault();
      showImage(index - 1);
    } else if (ev.key === 'ArrowRight') {
      ev.preventDefault();
      showImage(index + 1);
    }
  });

  showImage(0);
})();

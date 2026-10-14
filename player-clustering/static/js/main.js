/* static/js/main.js — Player Clustering Dashboard */

const $  = sel => document.querySelector(sel);
const $$ = sel => Array.from(document.querySelectorAll(sel));

const CLUSTER_COLORS = ['#c9714a', '#63c08a', '#6fa8dc', '#e5b45a', '#a98bd8', '#e2705f', '#4fb3a8', '#d98cc0'];

const state = {
  previewRows: [],
  previewCols: [],
  result: null,          // full result object from /api/analyze
  finalData: null,       // full row data for registry view
  registryFilters: { km: '', hc: '', text: '' },
  search: ''
};

/* ============================================================================
   THEME TOGGLE
   ============================================================================ */
$('#themeToggle').addEventListener('click', () => {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  toast('ok', 'Theme switched', `Now using ${next} mode.`);
});

/* ============================================================================
   NAVIGATION
   ============================================================================ */
const VIEW_META = {
  dashboard:  ['Dashboard', 'Cluster cricket players by performance using unsupervised ML'],
  datasource: ['Data Source', 'Upload a CSV or generate a synthetic sample dataset'],
  preprocess: ['Preprocessing', 'Cleaning, imputation and feature scaling'],
  kmeans:     ['K-Means Clustering', 'Elbow method, silhouette score and cluster profiles'],
  hier:       ['Hierarchical Clustering', 'Dendrogram with Ward linkage'],
  compare:    ['Comparison', 'PCA visualisation and Adjusted Rand Index'],
  registry:   ['Clustered Players', 'Final dataset with cluster assignments'],
  about:      ['About & Methods', 'ML concepts demonstrated in this project']
};

function switchView(name) {
  $$('.view').forEach(v => v.classList.remove('active'));
  const target = $(`#view-${name}`);
  if (target) target.classList.add('active');
  $$('.nav-item').forEach(n => n.classList.toggle('active', n.dataset.view === name));
  const [title, crumb] = VIEW_META[name] || ['', ''];
  $('#pageTitle').textContent = title;
  $('#pageCrumb').textContent = crumb;
}

$$('.nav-item').forEach(item => {
  item.addEventListener('click', () => switchView(item.dataset.view));
});

/* ============================================================================
   HELPERS
   ============================================================================ */
function esc(v) {
  if (v === null || v === undefined) return '—';
  return String(v).replace(/[&<>"']/g, c => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]));
}
function fmtNum(v, d = 2) {
  if (v === null || v === undefined || isNaN(v)) return '—';
  return Number(v).toFixed(d);
}
function toast(type, title, msg) {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `<div class="t-body"><strong>${esc(title)}</strong><span>${esc(msg)}</span></div>`;
  $('#toastWrap').appendChild(el);
  setTimeout(() => {
    el.style.transition = 'opacity .3s, transform .3s';
    el.style.opacity = '0';
    el.style.transform = 'translateX(30px)';
    setTimeout(() => el.remove(), 320);
  }, 3600);
}
function animateCount(el, target) {
  const start = parseInt(el.textContent, 10) || 0;
  const dur = 620;
  const t0 = performance.now();
  (function step(t) {
    const p = Math.min(1, (t - t0) / dur);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = Math.round(start + (target - start) * eased);
    if (p < 1) requestAnimationFrame(step);
  })(t0);
}
function setLoading(on) {
  $('#loading').classList.toggle('on', on);
}

/* ============================================================================
   DATA LOADING  (sample / upload)
   ============================================================================ */
$('#btnSample').addEventListener('click', async () => {
  setLoading(true);
  try {
    const res = await fetch('/api/sample', { method: 'POST' });
    const data = await res.json();
    setLoading(false);
    if (!res.ok) { toast('err', 'Sample failed', data.error || 'Unknown error'); return; }
    onDataLoaded(data, 'Sample dataset of 100 players generated.');
  } catch (err) {
    setLoading(false);
    toast('err', 'Network error', err.message);
  }
});

$('#btnUpload').addEventListener('click', async () => {
  const input = $('#fileInput');
  if (!input.files || !input.files.length) {
    toast('warn', 'No file selected', 'Please choose a CSV file first.');
    return;
  }
  setLoading(true);
  const formData = new FormData();
  formData.append('file', input.files[0]);
  try {
    const res = await fetch('/api/upload', { method: 'POST', body: formData });
    const data = await res.json();
    setLoading(false);
    if (!res.ok) { toast('err', 'Upload failed', data.error || 'Unknown error'); return; }
    onDataLoaded(data, 'CSV uploaded successfully.');
  } catch (err) {
    setLoading(false);
    toast('err', 'Network error', err.message);
  }
});

function onDataLoaded(data, message) {
  toast('ok', 'Data ready', message);
  state.previewRows = data.preview || [];
  state.previewCols = data.columns || (data.preview?.[0] ? Object.keys(data.preview[0]) : []);

  $('#statShape').textContent = `${data.shape[0]} rows × ${data.shape[1]} cols`;
  $('#statMissing').textContent = `${data.total_missing} missing cells`;
  $('#previewSubtitle').textContent = `First ${data.preview.length} of ${data.shape[0]} rows`;

  renderPreviewTable();
  $('#previewCard').style.display = '';
  $('#runAnalysis').disabled = false;

  // Sidebar badge
  $('#labelChip').textContent = `${data.shape[0]} players`;

  switchView('datasource');
}

/* Preview table with search */
$('#previewSearch').addEventListener('input', () => {
  state.search = $('#previewSearch').value.toLowerCase();
  renderPreviewTable();
});

function renderPreviewTable() {
  if (!state.previewRows.length) return;
  const cols = state.previewCols;

  $('#previewHead').innerHTML = cols.map(c => `<th>${esc(c)}</th>`).join('');

  const rows = state.previewRows.filter(r => {
    if (!state.search) return true;
    return cols.some(c => String(r[c]).toLowerCase().includes(state.search));
  });

  if (!rows.length) {
    $('#previewBody').innerHTML = `<tr><td colspan="${cols.length}"><div class="empty-state">No rows match the filter.</div></td></tr>`;
    return;
  }

  $('#previewBody').innerHTML = rows.map(r => `
    <tr>${cols.map(c => {
      const v = r[c];
      const isNum = typeof v === 'number';
      return `<td class="${isNum ? 'mono' : ''}">${esc(isNum ? fmtNum(v) : v)}</td>`;
    }).join('')}</tr>
  `).join('');
}

/* ============================================================================
   RUN ANALYSIS
   ============================================================================ */
$('#runAnalysis').addEventListener('click', runAnalysis);

async function runAnalysis() {
  const k = Math.max(2, Math.min(10, parseInt($('#kInput').value, 10) || 4));
  $('#kInput').value = k;

  setLoading(true);
  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ k })
    });
    const data = await res.json();
    setLoading(false);

    if (!res.ok) { toast('err', 'Analysis failed', data.error || 'Unknown error'); return; }

    state.result = data;
    renderAll(data);
    toast('ok', 'Analysis complete', `k=${data.k_used} · ${data.rows_used} players clustered.`);
    switchView('preprocess');
  } catch (err) {
    setLoading(false);
    toast('err', 'Network error', err.message);
  }
}

/* ============================================================================
   MASTER RENDER
   ============================================================================ */
function renderAll(d) {
  renderDashboard(d);
  renderPreprocessing(d);
  renderKMeans(d);
  renderHier(d);
  renderCompare(d);
  renderRegistry(d);
}

/* ---- 1. Dashboard ---- */
function renderDashboard(d) {
  animateCount($('#kpiPlayers'), d.rows_used);
  $('#kpiK').textContent = d.k_used;
  $('#kpiKFoot').textContent = `${d.k_used} clusters requested`;
  $('#kpiSil').textContent = Math.max(d.kmeans_silhouette, d.hier_silhouette).toFixed(3);
  $('#kpiAri').textContent = d.ari.toFixed(3);

  // Build the network from cluster profiles
  renderNetwork(d);

  // Cluster bars
  const maxSize = Math.max(...d.kmeans_profile.map(p => p.Size), 1);
  $('#clusterBars').innerHTML = d.kmeans_profile.map((p, i) => {
    const color = CLUSTER_COLORS[i % CLUSTER_COLORS.length];
    const label = d.kmeans_cluster_labels[String(p.Cluster)] || '—';
    return `
      <div style="margin-bottom:14px">
        <div style="display:flex;align-items:center;gap:8px;font-size:12.5px;margin-bottom:6px">
          <span style="width:9px;height:9px;border-radius:50%;background:${color};display:inline-block"></span>
          <b>Cluster ${p.Cluster} · ${esc(label)}</b>
          <span style="margin-left:auto;color:var(--muted);font-size:11.5px">${p.Size} players · Ø runs ${fmtNum(p.Runs, 0)} · Ø SR ${fmtNum(p.StrikeRate, 1)}</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width:${(p.Size / maxSize) * 100}%;background:${color}"></div>
        </div>
      </div>`;
  }).join('');

  // Method summary
  $('#methodSummary').innerHTML = `
    <div class="grid g-3">
      <div>
        <div class="stat-row"><span>Rows clustered</span><b>${d.rows_used}</b></div>
        <div class="stat-row"><span>Chosen k</span><b>${d.k_used}</b></div>
        <div class="stat-row"><span>Features used</span><b>6 numeric</b></div>
      </div>
      <div>
        <div class="stat-row"><span>K-Means inertia</span><b>${d.kmeans_inertia}</b></div>
        <div class="stat-row"><span>K-Means silhouette</span><b>${d.kmeans_silhouette}</b></div>
        <div class="stat-row"><span>K-Means clusters</span><b>${d.kmeans_profile.length}</b></div>
      </div>
      <div>
        <div class="stat-row"><span>Hier. silhouette</span><b>${d.hier_silhouette}</b></div>
        <div class="stat-row"><span>Hier. clusters</span><b>${d.hier_profile.length}</b></div>
        <div class="stat-row"><span>ARI</span><b style="color:var(--accent-2)">${d.ari}</b></div>
      </div>
    </div>`;
}

/* ---- Network visualisation (pure SVG, cluster constellation) ---- */
function renderNetwork(d) {
  const container = $('#dashNet');
  if (!d || !d.kmeans_profile) {
    container.innerHTML = '<div class="empty">No clusters yet</div>';
    return;
  }

  const W = 640, H = 340, pad = 60;
  const profiles = d.kmeans_profile;
  const nClusters = profiles.length;

  // Place cluster nodes on a circle
  const cx = W / 2, cy = H / 2;
  const R = Math.min(W, H) / 2 - pad;
  const clusterNodes = profiles.map((p, i) => {
    const angle = (i / nClusters) * Math.PI * 2 - Math.PI / 2;
    return {
      ...p,
      i,
      x: cx + Math.cos(angle) * R,
      y: cy + Math.sin(angle) * R,
      color: CLUSTER_COLORS[i % CLUSTER_COLORS.length],
      label: d.kmeans_cluster_labels[String(p.Cluster)] || `Cluster ${p.Cluster}`
    };
  });

  // Faint edges between cluster nodes (all pairs)
  const edges = [];
  for (let i = 0; i < clusterNodes.length; i++) {
    for (let j = i + 1; j < clusterNodes.length; j++) {
      edges.push(`<line x1="${clusterNodes[i].x}" y1="${clusterNodes[i].y}"
                       x2="${clusterNodes[j].x}" y2="${clusterNodes[j].y}"
                       stroke="var(--border)" stroke-width="1" stroke-dasharray="3 5" opacity="0.5"/>`);
    }
  }

  // Draw each cluster as a big node with N satellite dots around it
  const maxSize = Math.max(...clusterNodes.map(c => c.Size), 1);
  const satellites = clusterNodes.map(cn => {
    const n = Math.min(cn.Size, 14); // cap visible satellites
    const orbitR = 22 + (cn.Size / maxSize) * 18;
    const dots = [];
    for (let k = 0; k < n; k++) {
      const a = (k / n) * Math.PI * 2;
      const x = cn.x + Math.cos(a) * orbitR;
      const y = cn.y + Math.sin(a) * orbitR;
      dots.push(`<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2.4" fill="${cn.color}" opacity="0.6"/>`);
    }
    return dots.join('');
  }).join('');

  const bigNodes = clusterNodes.map(cn => `
    <g class="net-node" data-cluster="${cn.i}">
      <circle cx="${cn.x.toFixed(1)}" cy="${cn.y.toFixed(1)}" r="${(18 + (cn.Size / maxSize) * 12).toFixed(1)}"
              fill="${cn.color}" opacity="0.14"/>
      <circle cx="${cn.x.toFixed(1)}" cy="${cn.y.toFixed(1)}" r="${(11 + (cn.Size / maxSize) * 8).toFixed(1)}"
              fill="${cn.color}" stroke="var(--panel)" stroke-width="2.5" style="cursor:pointer"/>
      <text x="${cn.x.toFixed(1)}" y="${cn.y.toFixed(1)}" dy="4" text-anchor="middle"
            font-size="10.5" font-weight="700" fill="#fff" pointer-events="none">${cn.Size}</text>
      <text x="${cn.x.toFixed(1)}" y="${(cn.y + 34).toFixed(1)}" text-anchor="middle"
            font-size="10.5" font-weight="650" fill="${cn.color}" pointer-events="none">${esc(cn.label)}</text>
    </g>`).join('');

  container.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">
      ${edges.join('')}
      ${satellites}
      ${bigNodes}
    </svg>`;

  $('#netSummary').textContent = `${nClusters} clusters · ${d.rows_used} players`;

  // Hover tooltips
  container.querySelectorAll('.net-node').forEach(g => {
    const idx = +g.dataset.cluster;
    const cn = clusterNodes[idx];
    g.addEventListener('mouseenter', ev => {
      showTooltip(ev, cn, d);
    });
    g.addEventListener('mousemove', moveTooltip);
    g.addEventListener('mouseleave', hideTooltip);
  });
}

function showTooltip(ev, cn, d) {
  const tooltip = document.getElementById('mapTooltip') || createTooltipEl();
  tooltip.innerHTML = `
    <div style="font-weight:700;margin-bottom:6px;font-size:12.5px;color:${cn.color}">Cluster ${cn.Cluster} · ${esc(cn.label)}</div>
    <div style="display:flex;justify-content:space-between;gap:14px;font-size:11.5px;color:var(--muted)">
      <span>Players</span><b style="color:var(--text-dim)">${cn.Size}</b>
    </div>
    <div style="display:flex;justify-content:space-between;gap:14px;font-size:11.5px;color:var(--muted)">
      <span>Avg runs</span><b style="color:var(--text-dim)">${fmtNum(cn.Runs, 0)}</b>
    </div>
    <div style="display:flex;justify-content:space-between;gap:14px;font-size:11.5px;color:var(--muted)">
      <span>Avg wickets</span><b style="color:var(--text-dim)">${fmtNum(cn.Wickets, 0)}</b>
    </div>
    <div style="display:flex;justify-content:space-between;gap:14px;font-size:11.5px;color:var(--muted)">
      <span>Avg SR</span><b style="color:var(--text-dim)">${fmtNum(cn.StrikeRate, 1)}</b>
    </div>
    <div style="display:flex;justify-content:space-between;gap:14px;font-size:11.5px;color:var(--muted)">
      <span>Avg economy</span><b style="color:var(--text-dim)">${fmtNum(cn.EconomyRate, 2)}</b>
    </div>
  `;
  tooltip.classList.add('show');
  moveTooltip(ev);
}

function createTooltipEl() {
  const el = document.createElement('div');
  el.id = 'mapTooltip';
  el.style.cssText = `position:fixed;pointer-events:none;z-index:200;
    background:var(--panel-3);border:1px solid var(--border);border-radius:10px;
    padding:11px 14px;min-width:180px;box-shadow:0 14px 32px -12px rgba(0,0,0,.7);
    opacity:0;transform:translateY(4px);transition:opacity .12s,transform .12s;
    font-family:var(--sans);`;
  document.body.appendChild(el);
  return el;
}

function moveTooltip(ev) {
  const tt = document.getElementById('mapTooltip');
  if (!tt) return;
  tt.style.opacity = '1';
  tt.style.transform = 'none';
  const w = tt.offsetWidth, h = tt.offsetHeight;
  let x = ev.clientX + 16, y = ev.clientY + 16;
  if (x + w > window.innerWidth - 12) x = ev.clientX - w - 16;
  if (y + h > window.innerHeight - 12) y = ev.clientY - h - 16;
  tt.style.left = x + 'px';
  tt.style.top = y + 'px';
}

function hideTooltip() {
  const tt = document.getElementById('mapTooltip');
  if (tt) tt.style.opacity = '0';
}

/* ---- 2. Preprocessing ---- */
function renderPreprocessing(d) {
  const c = d.cleaning;
  $('#cleanReport').innerHTML = `
    <div class="stat-row"><span>Original rows</span><b>${c.original_rows}</b></div>
    <div class="stat-row"><span>Duplicate rows removed</span><b style="color:var(--amber)">${c.duplicates_removed}</b></div>
    <div class="stat-row"><span>Unusable rows dropped</span><b style="color:var(--amber)">${c.rows_dropped}</b></div>
    <div class="stat-row"><span>Missing cells filled (median)</span><b style="color:var(--amber)">${c.missing_filled}</b></div>
    <div class="stat-row"><span>Rows finally clustered</span><b style="color:var(--green)">${c.final_rows}</b></div>
    <div style="margin-top:14px;font-size:11.5px;color:var(--muted);line-height:1.7">
      <b style="color:var(--text-dim)">Cricket-aware rule:</b> a blank <em>EconomyRate</em> paired with
      zero <em>Wickets</em> means "did not bowl" — we fill those with 0, not the median, so a pure
      batter isn't credited with an average bowling economy.
    </div>`;

  const rows = d.scaling.map(r => `
    <tr>
      <td><b>${esc(r.feature)}</b></td>
      <td class="mono">${r.before_mean}</td>
      <td class="mono">${r.before_std}</td>
      <td class="mono" style="color:var(--green)">${r.after_mean}</td>
      <td class="mono" style="color:var(--green)">${r.after_std}</td>
    </tr>`).join('');

  $('#scaleReport').innerHTML = `
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>Feature</th><th>Before mean</th><th>Before std</th>
          <th>After mean</th><th>After std</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    <p style="font-size:11.5px;color:var(--muted);padding:12px 14px 4px;line-height:1.7">
      After standardisation every column has mean ≈ 0 and std ≈ 1, so no single feature
      can dominate the Euclidean distance calculations inside K-Means or Ward linkage.
    </p>`;
}

/* ---- 3. K-Means ---- */
function renderKMeans(d) {
  $('#kmeansChartFrame').innerHTML = `
    <img src="data:image/png;base64,${d.elbow_img}" alt="Elbow & Silhouette charts" />
    <div class="chart-toolbar">
      <a class="btn ghost" download="elbow_silhouette.png"
         href="data:image/png;base64,${d.elbow_img}">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 3v12M7 10l5 5 5-5M4 20h16"/></svg>
        Save chart
      </a>
    </div>`;

  const numericCols = ['Matches','Runs','BattingAverage','StrikeRate','Wickets','EconomyRate'];
  const rows = d.kmeans_profile.map(p => {
    const label = d.kmeans_cluster_labels[String(p.Cluster)] || '—';
    const color = CLUSTER_COLORS[p.Cluster % CLUSTER_COLORS.length];
    return `<tr>
      <td><b style="color:${color}">●</b> ${p.Cluster}</td>
      <td>${p.Size}</td>
      ${numericCols.map(c => `<td class="mono">${fmtNum(p[c])}</td>`).join('')}
      <td><span class="chip" style="background:${color}22;color:${color};border-color:${color}55">${esc(label)}</span></td>
    </tr>`;
  }).join('');

  $('#kmeansProfile').innerHTML = `
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>Cluster</th><th>Size</th>
          ${numericCols.map(c => `<th>${esc(c)}</th>`).join('')}
          <th>Auto label</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;

  $('#kmInertia').textContent = d.kmeans_inertia;
  $('#kmSil').textContent = d.kmeans_silhouette.toFixed(3);
  $('#kmSize').textContent = d.rows_used;
}

/* ---- 4. Hierarchical ---- */
function renderHier(d) {
  $('#hierChartFrame').innerHTML = `
    <img src="data:image/png;base64,${d.dendro_img}" alt="Dendrogram" />
    <div class="chart-toolbar">
      <a class="btn ghost" download="dendrogram.png"
         href="data:image/png;base64,${d.dendro_img}">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 3v12M7 10l5 5 5-5M4 20h16"/></svg>
        Save chart
      </a>
    </div>`;

  const numericCols = ['Matches','Runs','BattingAverage','StrikeRate','Wickets','EconomyRate'];
  const rows = d.hier_profile.map(p => {
    const label = d.hier_cluster_labels[String(p.Cluster)] || '—';
    const color = CLUSTER_COLORS[p.Cluster % CLUSTER_COLORS.length];
    return `<tr>
      <td><b style="color:${color}">●</b> ${p.Cluster}</td>
      <td>${p.Size}</td>
      ${numericCols.map(c => `<td class="mono">${fmtNum(p[c])}</td>`).join('')}
      <td><span class="chip" style="background:${color}22;color:${color};border-color:${color}55">${esc(label)}</span></td>
    </tr>`;
  }).join('');

  $('#hierProfile').innerHTML = `
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>Cluster</th><th>Size</th>
          ${numericCols.map(c => `<th>${esc(c)}</th>`).join('')}
          <th>Auto label</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

/* ---- 5. Compare ---- */
function renderCompare(d) {
  $('#pcaFrame').innerHTML = `
    <img src="data:image/png;base64,${d.pca_img}" alt="PCA scatter comparison" />
    <div class="chart-toolbar">
      <a class="btn ghost" download="pca_scatter.png"
         href="data:image/png;base64,${d.pca_img}">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 3v12M7 10l5 5 5-5M4 20h16"/></svg>
        Save chart
      </a>
    </div>
    <p style="font-size:11.5px;color:var(--muted);margin-top:10px">
      PC1 explains ${d.pca_explained[0]}% of variance · PC2 explains ${d.pca_explained[1]}%.
      Together they capture the dominant structure of the 6-feature space.
    </p>`;

  $('#ariBig').textContent = d.ari.toFixed(3);
  $('#ariSentence').textContent = d.ari_sentence;
  $('#ariKmInertia').textContent = d.kmeans_inertia;
  $('#ariKmSil').textContent = d.kmeans_silhouette.toFixed(3);
  $('#ariHcSil').textContent = d.hier_silhouette.toFixed(3);

  $('#labelLogicSubtitle').textContent = 'Rule-based labels derived from cluster centroid averages';
  $('#labelLogic').innerHTML = `
    <div style="font-size:13px;color:var(--text-dim);line-height:1.8;margin-bottom:18px">
      ${esc(d.label_logic)}
    </div>
    <div class="grid g-2">
      <div>
        <h4 style="font-size:12.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.9px;margin-bottom:10px">K-Means labels</h4>
        ${Object.entries(d.kmeans_cluster_labels).sort((a,b)=>Number(a[0])-Number(b[0])).map(([cid, label]) => {
          const color = CLUSTER_COLORS[Number(cid) % CLUSTER_COLORS.length];
          return `<div class="stat-row"><span>Cluster ${cid}</span><b style="color:${color}">${esc(label)}</b></div>`;
        }).join('')}
      </div>
      <div>
        <h4 style="font-size:12.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.9px;margin-bottom:10px">Hierarchical labels</h4>
        ${Object.entries(d.hier_cluster_labels).sort((a,b)=>Number(a[0])-Number(b[0])).map(([cid, label]) => {
          const color = CLUSTER_COLORS[Number(cid) % CLUSTER_COLORS.length];
          return `<div class="stat-row"><span>Cluster ${cid}</span><b style="color:${color}">${esc(label)}</b></div>`;
        }).join('')}
      </div>
    </div>`;
}

/* ---- 6. Registry ---- */
async function fetchFinalData() {
  // The backend has already written the clustered CSV. Fetch it and parse.
  try {
    const res = await fetch('/download');
    if (!res.ok) return null;
    const text = await res.text();
    return parseCSV(text);
  } catch {
    return null;
  }
}

function parseCSV(text) {
  const lines = text.split(/\r?\n/).filter(l => l.length);
  if (!lines.length) return null;
  const header = splitCSVLine(lines[0]);
  const rows = lines.slice(1).map(line => {
    const cells = splitCSVLine(line);
    const obj = {};
    header.forEach((h, i) => { obj[h] = cells[i]; });
    return obj;
  });
  return { header, rows };
}

function splitCSVLine(line) {
  const out = [];
  let cur = '', inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') { cur += '"'; i++; }
      else inQuotes = !inQuotes;
    } else if (ch === ',' && !inQuotes) {
      out.push(cur); cur = '';
    } else {
      cur += ch;
    }
  }
  out.push(cur);
  return out;
}

async function renderRegistry(d) {
  $('#registrySubtitle').textContent = `${d.rows_used} players with K-Means and Hierarchical labels`;

  const parsed = await fetchFinalData();
  if (!parsed) {
    $('#registryBody').innerHTML = `<tr><td colspan="20"><div class="empty-state">Could not load the clustered dataset.</div></td></tr>`;
    return;
  }
  state.finalData = parsed;

  const head = parsed.header;
  $('#registryHead').innerHTML = head.map(h => `<th>${esc(h)}</th>`).join('');

  // Populate filters
  const kmLabels = [...new Set(parsed.rows.map(r => r.KMeans_Label))].filter(Boolean).sort();
  const hcLabels = [...new Set(parsed.rows.map(r => r.Hierarchical_Label))].filter(Boolean).sort();
  $('#filterKm').innerHTML = `<option value="">All K-Means labels</option>` +
    kmLabels.map(l => `<option value="${esc(l)}">${esc(l)}</option>`).join('');
  $('#filterHc').innerHTML = `<option value="">All Hierarchical labels</option>` +
    hcLabels.map(l => `<option value="${esc(l)}">${esc(l)}</option>`).join('');

  renderRegistryRows();
}

function renderRegistryRows() {
  if (!state.finalData) return;
  const { header, rows } = state.finalData;
  const f = state.registryFilters;

  let filtered = rows;
  if (f.km) filtered = filtered.filter(r => r.KMeans_Label === f.km);
  if (f.hc) filtered = filtered.filter(r => r.Hierarchical_Label === f.hc);
  if (f.text) {
    const q = f.text.toLowerCase();
    filtered = filtered.filter(r => Object.values(r).some(v => String(v).toLowerCase().includes(q)));
  }

  $('#filteredCount').textContent = `${filtered.length} / ${rows.length} players`;

  if (!filtered.length) {
    $('#registryBody').innerHTML = `<tr><td colspan="${header.length}"><div class="empty-state">No players match the filters.</div></td></tr>`;
    return;
  }

  $('#registryBody').innerHTML = filtered.map(r => {
    const cells = header.map(h => {
      const v = r[h];
      if (h === 'KMeans_Label' || h === 'Hierarchical_Label') {
        const idx = header.indexOf(h) === 5 ? r.KMeans_Cluster : r.Hierarchical_Cluster;
        const color = CLUSTER_COLORS[Number(idx) % CLUSTER_COLORS.length] || '#888';
        return `<td><span class="chip" style="background:${color}22;color:${color};border-color:${color}55">${esc(v)}</span></td>`;
      }
      if (h.endsWith('_Cluster')) {
        return `<td class="mono">${esc(v)}</td>`;
      }
      const numeric = !isNaN(parseFloat(v)) && isFinite(v) && !isNaN(v);
      return `<td class="${numeric ? 'mono' : ''}">${esc(numeric ? fmtNum(v) : v)}</td>`;
    }).join('');
    return `<tr>${cells}</tr>`;
  }).join('');
}

$('#filterKm').addEventListener('change', e => { state.registryFilters.km = e.target.value; renderRegistryRows(); });
$('#filterHc').addEventListener('change', e => { state.registryFilters.hc = e.target.value; renderRegistryRows(); });
$('#filterPlayer').addEventListener('input', e => { state.registryFilters.text = e.target.value; renderRegistryRows(); });

/* Global search → jump to registry */
$('#globalSearch').addEventListener('input', e => {
  const q = e.target.value;
  if (q && $('#view-registry').classList.contains('active')) {
    state.registryFilters.text = q;
    $('#filterPlayer').value = q;
    renderRegistryRows();
  }
});

/* Export button */
$('#exportBtn').addEventListener('click', () => {
  if (!state.result) {
    toast('warn', 'No results yet', 'Run the analysis before downloading.');
    return;
  }
  window.location.href = '/download';
  toast('ok', 'Download started', 'Your clustered CSV is being prepared.');
});

/* ============================================================================
   BOOTSTRAP
   ============================================================================ */
(function init() {
  // Set initial view
  switchView('dashboard');
  // Set the dropdown filter placeholder states
  toast('ok', 'Dashboard ready', 'Load a dataset from the Data Source tab to begin.');
})();

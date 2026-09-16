(function () {
  "use strict";

  const DATASETS = window.RESULTS_DATA || {};
  const METRICS = [
    "BERTScore_f1",
    "BERTScore_precision",
    "BERTScore_recall",
    "BLEU",
    "Cosine",
    "JSD",
    "Jaccard",
    "Levenshtein",
    "ROUGE_rouge1",
    "ROUGE_rouge2",
    "ROUGE_rougeL",
    "SequenceMatcher"
  ];
  const NUMERIC_COLUMNS = new Set(["_compositeScore", ...METRICS]);
  const FILTER_COLUMNS = ["model_name", "uses_metadata", "prompt", "epochs", "sft_size", "training_cfg", "lora_rank"];
  const TRAINING_COLUMNS = new Set(["epochs", "sft_size", "training_cfg", "lora_rank"]);
  const COLORS = ["#157f88", "#3466b3", "#c85f4c", "#b8871b", "#4d8b4a", "#6b5aa6", "#8a6d3b"];
  const COLUMN_DEFS = [
    ["_compositeScore", "Composite"],
    ["model_name", "Model"],
    ["uses_metadata", "Metadata"],
    ["prompt", "Prompt"],
    ["epochs", "Epochs"],
    ["sft_size", "SFT size"],
    ["training_cfg", "Config"],
    ["lora_rank", "LoRA"],
    ["_isBaseline", "Run type"],
    ...METRICS.map((metric) => [metric, metric])
  ].map(([key, label]) => ({ key, label }));

  const state = {
    activeDataset: Object.keys(DATASETS)[0],
    filters: Object.fromEntries(FILTER_COLUMNS.map((key) => [key, key === "model_name" ? "M13" : "__all__"])),
    runType: "all",
    search: "",
    metric: "BERTScore_f1",
    sortKey: "_compositeScore",
    sortDirection: "desc",
    selectedRowId: null,
    visibleColumns: new Set(COLUMN_DEFS.map((column) => column.key))
  };

  const els = {
    datasetTabs: document.getElementById("datasetTabs"),
    summaryGrid: document.getElementById("summaryGrid"),
    searchInput: document.getElementById("searchInput"),
    runTypeFilter: document.getElementById("runTypeFilter"),
    resetButton: document.getElementById("resetButton"),
    exportButton: document.getElementById("exportButton"),
    columnControls: document.getElementById("columnControls"),
    topBars: document.getElementById("topBars"),
    heatmap: document.getElementById("heatmap"),
    detailPanel: document.getElementById("detailPanel"),
    table: document.getElementById("resultsTable"),
    tableMeta: document.getElementById("tableMeta"),
    selectedMeta: document.getElementById("selectedMeta"),
    topChartMeta: document.getElementById("topChartMeta"),
    heatmapMeta: document.getElementById("heatmapMeta")
  };

  function init() {
    if (!state.activeDataset) {
      document.body.innerHTML = "<main class=\"shell\"><div class=\"empty-state\">No datasets found.</div></main>";
      return;
    }
    renderDatasetTabs();
    renderFilterOptions();
    renderColumnControls();
    bindControls();
    render();
  }

  function bindControls() {
    els.searchInput.addEventListener("input", () => {
      state.search = els.searchInput.value.trim().toLowerCase();
      render();
    });
    els.runTypeFilter.addEventListener("change", () => {
      state.runType = els.runTypeFilter.value;
      render();
    });
    els.resetButton.addEventListener("click", () => {
      state.filters = Object.fromEntries(FILTER_COLUMNS.map((key) => [key, key === "model_name" ? "M13" : "__all__"]));
      state.runType = "all";
      state.search = "";
      state.sortKey = "_compositeScore";
      state.sortDirection = "desc";
      state.selectedRowId = null;
      els.searchInput.value = "";
      els.runTypeFilter.value = "all";
      renderFilterOptions();
      render();
    });
    els.exportButton.addEventListener("click", exportFilteredRows);

    document.querySelectorAll("[data-filter]").forEach((select) => {
      select.addEventListener("change", () => {
        state.filters[select.dataset.filter] = select.value;
        render();
      });
    });
  }

  function rowsForDataset() {
    return DATASETS[state.activeDataset].rows || [];
  }

  function renderDatasetTabs() {
    els.datasetTabs.innerHTML = Object.entries(DATASETS).map(([key, dataset]) => {
      const selected = key === state.activeDataset ? "true" : "false";
      return `<button type="button" data-dataset="${escapeHTML(key)}" aria-selected="${selected}">${escapeHTML(dataset.label)}</button>`;
    }).join("");
    els.datasetTabs.querySelectorAll("button").forEach((button) => {
      button.addEventListener("click", () => {
        state.activeDataset = button.dataset.dataset;
        state.selectedRowId = null;
        renderDatasetTabs();
        renderFilterOptions();
        render();
      });
    });
  }

  function renderFilterOptions() {
    const rows = rowsForDataset();
    FILTER_COLUMNS.forEach((key) => {
      const select = document.querySelector(`[data-filter="${key}"]`);
      const values = uniqueValues(rows, key);
      const previous = state.filters[key];
      const options = [`<option value="__all__">All</option>`].concat(values.map((value) => {
        return `<option value="${escapeAttr(value)}">${escapeHTML(displayValue(key, value))}</option>`;
      }));
      select.innerHTML = options.join("");
      state.filters[key] = values.includes(previous) ? previous : "__all__";
      select.value = state.filters[key];
    });
  }

  function renderColumnControls() {
    els.columnControls.innerHTML = COLUMN_DEFS.map((column) => {
      const checked = state.visibleColumns.has(column.key) ? "checked" : "";
      return `<label><input type="checkbox" value="${escapeAttr(column.key)}" ${checked}> ${escapeHTML(column.label)}</label>`;
    }).join("");
    els.columnControls.querySelectorAll("input").forEach((input) => {
      input.addEventListener("change", () => {
        if (input.checked) {
          state.visibleColumns.add(input.value);
        } else if (state.visibleColumns.size > 1) {
          state.visibleColumns.delete(input.value);
        } else {
          input.checked = true;
        }
        renderTable(getSortedRows(getFilteredRows()));
      });
    });
  }

  function render() {
    const filteredRows = getFilteredRows();
    const sortedRows = getSortedRows(filteredRows);
    if (!sortedRows.some((row) => row._rowId === state.selectedRowId)) {
      state.selectedRowId = sortedRows[0] ? sortedRows[0]._rowId : null;
    }
    renderSummary(filteredRows, sortedRows);
    renderTopBars(filteredRows);
    renderHeatmap(filteredRows);
    renderDetail(sortedRows);
    renderTable(sortedRows);
  }

  function getFilteredRows() {
    return rowsForDataset().filter((row) => {
      for (const key of FILTER_COLUMNS) {
        if (state.filters[key] !== "__all__" && filterValue(row, key) !== state.filters[key]) {
          return false;
        }
      }
      if (state.runType === "baseline" && !row._isBaseline) {
        return false;
      }
      if (state.runType === "trained" && row._isBaseline) {
        return false;
      }
      if (state.search) {
        const haystack = COLUMN_DEFS.map((column) => displayValue(column.key, filterValue(row, column.key))).join(" ").toLowerCase();
        if (!haystack.includes(state.search)) {
          return false;
        }
      }
      return true;
    });
  }

  function getSortedRows(rows) {
    const direction = state.sortDirection === "asc" ? 1 : -1;
    return [...rows].sort((a, b) => {
      const aValue = a[state.sortKey];
      const bValue = b[state.sortKey];
      if (NUMERIC_COLUMNS.has(state.sortKey)) {
        const aNum = Number.isFinite(aValue) ? aValue : -Infinity;
        const bNum = Number.isFinite(bValue) ? bValue : -Infinity;
        return (aNum - bNum) * direction;
      }
      return String(displayValue(state.sortKey, aValue)).localeCompare(String(displayValue(state.sortKey, bValue))) * direction;
    });
  }

  function renderSummary(filteredRows, sortedRows) {
    const bestRow = [...filteredRows].sort((a, b) => b._compositeScore - a._compositeScore)[0];
    const bestModel = bestGroup(filteredRows, modelVariantLabel, "_compositeScore");
    const bestPrompt = bestGroup(filteredRows, "prompt", "_compositeScore");
    const dataset = DATASETS[state.activeDataset];
    const cards = [
      ["Best row", bestRow ? formatPercent(bestRow._compositeScore) : "None", bestRow ? configLabel(bestRow) : "No matching rows"],
      ["Best model", bestModel ? bestModel.key : "None", bestModel ? formatPercent(bestModel.value) : "No matching rows"],
      ["Best prompt", bestPrompt ? bestPrompt.key : "None", bestPrompt ? formatPercent(bestPrompt.value) : "No matching rows"],
      ["Rows", `${filteredRows.length} / ${rowsForDataset().length}`, "Filtered / total"],
      ["Dataset", dataset.label, dataset.sourceFile]
    ];
    els.summaryGrid.innerHTML = cards.map(([label, value, detail]) => {
      return `<article class="summary-card"><span>${escapeHTML(label)}</span><strong>${escapeHTML(value)}</strong><small>${escapeHTML(detail)}</small></article>`;
    }).join("");
    els.tableMeta.textContent = `${sortedRows.length} rows sorted by ${columnLabel(state.sortKey)} ${state.sortDirection}`;
  }

  function renderTopBars(rows) {
    const topRows = [...rows].sort((a, b) => b._compositeScore - a._compositeScore).slice(0, 12);
    els.topChartMeta.textContent = "Composite score";
    if (!topRows.length) {
      els.topBars.innerHTML = emptyState("No rows match the current filters.");
      return;
    }
    const rowHeight = 34;
    const width = 1040;
    const height = 48 + topRows.length * rowHeight;
    const margin = { left: 280, right: 70, top: 10, bottom: 34 };
    const innerWidth = width - margin.left - margin.right;
    const modelColors = colorMap(rows.map(modelVariantLabel));
    const bars = topRows.map((row, index) => {
      const y = margin.top + index * rowHeight + 6;
      const barWidth = Math.max(2, row._compositeScore * innerWidth);
      const label = truncate(configLabel(row), 48);
      return `
        <text class="chart-label" x="${margin.left - 12}" y="${y + 16}" text-anchor="end">${escapeHTML(label)}</text>
        <rect x="${margin.left}" y="${y}" width="${barWidth}" height="20" rx="5" fill="${modelColors.get(modelVariantLabel(row))}"></rect>
        <text class="value-text" x="${margin.left + barWidth + 8}" y="${y + 15}">${formatPercent(row._compositeScore)}</text>
      `;
    }).join("");
    els.topBars.innerHTML = `
      <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Top configurations by composite score">
        <line x1="${margin.left}" y1="${height - 24}" x2="${width - margin.right}" y2="${height - 24}" stroke="#d7e0e4"></line>
        <text class="axis-text" x="${margin.left}" y="${height - 6}">0</text>
        <text class="axis-text" x="${width - margin.right}" y="${height - 6}" text-anchor="end">1.0</text>
        ${bars}
      </svg>
    `;
  }

  function renderHeatmap(rows) {
    const metric = state.metric;
    els.heatmapMeta.textContent = `${metric} avg`;
    const models = uniqueModelVariants(rows);
    const prompts = uniqueValues(rows, "prompt");
    if (!models.length || !prompts.length) {
      els.heatmap.innerHTML = emptyState("No rows match the current filters.");
      return;
    }
    const values = [];
    const lookup = new Map();
    models.forEach((model) => {
      prompts.forEach((prompt) => {
        const subset = rows.filter((row) => modelVariantLabel(row) === model && row.prompt === prompt);
        const value = average(subset, metric);
        if (Number.isFinite(value)) {
          values.push(value);
          lookup.set(`${model}||${prompt}`, value);
        }
      });
    });
    const min = Math.min(...values);
    const max = Math.max(...values);
    const heads = [`<div class="heatmap-head">Model</div>`].concat(prompts.map((prompt) => {
      return `<div class="heatmap-head">${escapeHTML(prompt)}</div>`;
    })).join("");
    const cells = models.map((model) => {
      const rowHead = `<div class="heatmap-head">${escapeHTML(model)}</div>`;
      const rowCells = prompts.map((prompt) => {
        const value = lookup.get(`${model}||${prompt}`);
        if (!Number.isFinite(value)) {
          return `<div class="heatmap-cell empty">none</div>`;
        }
        const t = max === min ? 0.5 : (value - min) / (max - min);
        const bg = heatColor(t);
        const fg = t > 0.62 ? "#ffffff" : "#1e2a2f";
        return `<div class="heatmap-cell" style="background:${bg};color:${fg}" title="${escapeAttr(model)} ${escapeAttr(prompt)}">${formatMetric(value)}</div>`;
      }).join("");
      return rowHead + rowCells;
    }).join("");
    els.heatmap.innerHTML = `<div class="heatmap-grid" style="grid-template-columns: 120px repeat(${prompts.length}, minmax(78px, 1fr));">${heads}${cells}</div>`;
  }

  function renderDetail(rows) {
    const row = rows.find((candidate) => candidate._rowId === state.selectedRowId);
    els.selectedMeta.textContent = row ? row._rowId : "";
    if (!row) {
      els.detailPanel.innerHTML = emptyState("Select a row from the table.");
      return;
    }
    const keys = ["model_name", "uses_metadata", "prompt", "epochs", "sft_size", "training_cfg", "lora_rank", "_isBaseline", "_compositeScore", ...METRICS];
    els.detailPanel.innerHTML = keys.map((key) => {
      return `<div class="detail-item"><span>${escapeHTML(columnLabel(key))}</span><strong>${formatCell(key, row[key])}</strong></div>`;
    }).join("");
  }

  function renderTable(rows) {
    const visible = COLUMN_DEFS.filter((column) => state.visibleColumns.has(column.key));
    const head = visible.map((column) => {
      const active = column.key === state.sortKey ? ` (${state.sortDirection})` : "";
      return `<th><button type="button" data-sort="${escapeAttr(column.key)}">${escapeHTML(column.label)}${escapeHTML(active)}</button></th>`;
    }).join("");
    const body = rows.map((row) => {
      const selected = row._rowId === state.selectedRowId ? "selected" : "";
      const cells = visible.map((column) => {
        const numeric = NUMERIC_COLUMNS.has(column.key) ? " class=\"numeric\"" : "";
        return `<td${numeric}>${formatCell(column.key, row[column.key])}</td>`;
      }).join("");
      return `<tr class="${selected}" data-row-id="${escapeAttr(row._rowId)}">${cells}</tr>`;
    }).join("");
    els.table.innerHTML = `<thead><tr>${head}</tr></thead><tbody>${body}</tbody>`;
    els.table.querySelectorAll("th button").forEach((button) => {
      button.addEventListener("click", () => {
        const key = button.dataset.sort;
        if (state.sortKey === key) {
          state.sortDirection = state.sortDirection === "desc" ? "asc" : "desc";
        } else {
          state.sortKey = key;
          state.sortDirection = NUMERIC_COLUMNS.has(key) ? "desc" : "asc";
        }
        render();
      });
    });
    els.table.querySelectorAll("tbody tr").forEach((tr) => {
      tr.addEventListener("click", () => {
        state.selectedRowId = tr.dataset.rowId;
        renderDetail(rows);
        renderTable(rows);
      });
    });
  }

  function exportFilteredRows() {
    const rows = getSortedRows(getFilteredRows());
    const visible = COLUMN_DEFS.filter((column) => state.visibleColumns.has(column.key));
    const lines = [visible.map((column) => csvEscape(column.label)).join(",")];
    rows.forEach((row) => {
      lines.push(visible.map((column) => csvEscape(exportValue(column.key, row[column.key]))).join(","));
    });
    const blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${state.activeDataset}_filtered_results.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  function bestGroup(rows, groupKey, metric) {
    return groupedAverage(rows, groupKey, metric).sort((a, b) => b.value - a.value)[0] || null;
  }

  function groupedAverage(rows, groupKey, metric) {
    const groups = new Map();
    rows.forEach((row) => {
      const value = row[metric];
      if (!Number.isFinite(value)) {
        return;
      }
      const key = typeof groupKey === "function" ? groupKey(row) : filterValue(row, groupKey);
      const groupId = key || "baseline/none";
      const group = groups.get(groupId) || { key: groupId, total: 0, count: 0 };
      group.total += value;
      group.count += 1;
      groups.set(groupId, group);
    });
    return [...groups.values()].map((group) => ({ key: group.key, value: group.total / group.count, count: group.count }));
  }

  function average(rows, metric) {
    const values = rows.map((row) => row[metric]).filter(Number.isFinite);
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : NaN;
  }

  function uniqueValues(rows, key) {
    return [...new Set(rows.map((row) => filterValue(row, key)))].sort((a, b) => {
      if (a === "") return 1;
      if (b === "") return -1;
      return String(a).localeCompare(String(b), undefined, { numeric: true });
    });
  }

  function uniqueModelVariants(rows) {
    return [...new Set(rows.map(modelVariantLabel))].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  }

  function modelVariantLabel(row) {
    const model = row.model_name || "none";
    return usesMetadataValue(row.uses_metadata) ? `${model} + metadata` : model;
  }

  function filterValue(row, key) {
    if (key === "uses_metadata") {
      return usesMetadataValue(row.uses_metadata) ? "true" : "false";
    }
    const value = row[key];
    return value === undefined || value === null ? "" : value;
  }

  function usesMetadataValue(value) {
    if (typeof value === "boolean") {
      return value;
    }
    return ["true", "1", "yes"].includes(String(value ?? "").trim().toLowerCase());
  }

  function colorMap(values) {
    const unique = [...new Set(values)].sort();
    return new Map(unique.map((value, index) => [value, COLORS[index % COLORS.length]]));
  }

  function configLabel(row) {
    const parts = [modelVariantLabel(row), row.prompt];
    const config = [row.epochs, row.sft_size, row.training_cfg, row.lora_rank].filter(Boolean).join("/");
    parts.push(config || "baseline/none");
    return parts.filter(Boolean).join(" | ");
  }

  function displayValue(key, value) {
    if (key === "uses_metadata") {
      return usesMetadataValue(value) ? "Uses metadata" : "No metadata";
    }
    if (key === "_isBaseline") {
      return value ? "Baseline" : "Trained";
    }
    if ((value === "" || value === null || value === undefined) && TRAINING_COLUMNS.has(key)) {
      return "baseline/none";
    }
    if (value === "" || value === null || value === undefined) {
      return "none";
    }
    return value;
  }

  function formatCell(key, value) {
    if (key === "uses_metadata") {
      const enabled = usesMetadataValue(value);
      const label = enabled ? "Uses metadata" : "No metadata";
      const cls = enabled ? "pill metadata" : "pill";
      return `<span class="${cls}">${label}</span>`;
    }
    if (key === "_isBaseline") {
      const label = value ? "Baseline" : "Trained";
      const cls = value ? "pill baseline" : "pill";
      return `<span class="${cls}">${label}</span>`;
    }
    if (key === "_compositeScore") {
      return formatPercent(value);
    }
    if (NUMERIC_COLUMNS.has(key)) {
      return formatMetric(value);
    }
    return escapeHTML(displayValue(key, value));
  }

  function exportValue(key, value) {
    if (key === "uses_metadata") {
      return usesMetadataValue(value) ? "Uses metadata" : "No metadata";
    }
    if (key === "_isBaseline") {
      return value ? "Baseline" : "Trained";
    }
    if (NUMERIC_COLUMNS.has(key)) {
      return Number.isFinite(value) ? String(value) : "";
    }
    return displayValue(key, value);
  }

  function columnLabel(key) {
    const column = COLUMN_DEFS.find((candidate) => candidate.key === key);
    return column ? column.label : key;
  }

  function formatMetric(value) {
    if (!Number.isFinite(value)) {
      return "none";
    }
    return value >= 10 ? value.toFixed(1) : value.toFixed(4);
  }

  function formatPercent(value) {
    if (!Number.isFinite(value)) {
      return "none";
    }
    return value.toFixed(3);
  }

  function truncate(value, max) {
    return value.length > max ? `${value.slice(0, max - 3)}...` : value;
  }

  function emptyState(message) {
    return `<div class="empty-state">${escapeHTML(message)}</div>`;
  }

  function heatColor(t) {
    const low = [238, 243, 245];
    const mid = [242, 198, 109];
    const high = [21, 127, 136];
    const from = t < 0.5 ? low : mid;
    const to = t < 0.5 ? mid : high;
    const local = t < 0.5 ? t * 2 : (t - 0.5) * 2;
    const rgb = from.map((start, index) => Math.round(start + (to[index] - start) * local));
    return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
  }

  function csvEscape(value) {
    const text = String(value ?? "");
    if (/[",\n]/.test(text)) {
      return `"${text.replace(/"/g, "\"\"")}"`;
    }
    return text;
  }

  function escapeHTML(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      "\"": "&quot;",
      "'": "&#39;"
    })[char]);
  }

  function escapeAttr(value) {
    return escapeHTML(value);
  }

  init();
})();

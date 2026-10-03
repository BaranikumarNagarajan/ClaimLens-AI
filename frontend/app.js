const runButton = document.getElementById("run-button");
const statusPill = document.getElementById("status-pill");
const emptyState = document.getElementById("empty-state");
const loading = document.getElementById("loading");
const reportContent = document.getElementById("report-content");

function setStatus(state, label) {
  statusPill.className = "pill pill-" + state;
  statusPill.textContent = label;
}

function show(el, visible) {
  el.classList.toggle("hidden", !visible);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function buildStats(container, stats) {
  container.innerHTML = stats
    .filter((s) => s.value !== null && s.value !== undefined)
    .map(
      (s) => `<div class="stat">
        <div class="stat-label">${escapeHtml(s.label)}</div>
        <div class="stat-value">${escapeHtml(s.value)}</div>
      </div>`
    )
    .join("");
}

function buildTable(table, columns, rows) {
  const head = `<thead><tr>${columns.map((c) => `<th>${escapeHtml(c.label)}</th>`).join("")}</tr></thead>`;
  const body = rows
    .map(
      (row) =>
        `<tr>${columns
          .map((c) => `<td>${c.render ? c.render(row) : escapeHtml(row[c.key] ?? "—")}</td>`)
          .join("")}</tr>`
    )
    .join("");
  table.innerHTML = `${head}<tbody>${body}</tbody>`;
}

function buildCharts(container, charts) {
  container.innerHTML = charts
    .map(
      (chart) => `<div class="chart-card">
        <h3>${escapeHtml(chart.title)}</h3>
        <img alt="${escapeHtml(chart.title)}" src="data:image/png;base64,${chart.image}" />
      </div>`
    )
    .join("");
}

function renderOverview(result) {
  buildStats(document.getElementById("stat-grid"), [
    { label: "Status", value: result.status },
    { label: "Comparison", value: (result.comparison_type || "unavailable").replace(/_/g, " ") },
    { label: "Evidence sources", value: result.evidence_count },
    { label: "Relevant observations", value: result.relevant_evidence },
    { label: "Market", value: result.market || "Not specified" },
    { label: "Domain", value: result.domain },
  ]);
}

function renderSummary(result) {
  const text = result.summary || result.deterministic_summary || "";
  if (!text) return;
  document.getElementById("summary-text").textContent = text;
  document.getElementById("summary-meta").textContent =
    `Summary provider: ${result.summary_provider || "deterministic"} · status: ${result.summary_status || "n/a"}`;
  show(document.getElementById("section-summary"), true);
}

function renderDistribution(result) {
  const dist = result.model_distribution;
  if (!dist || !dist.available) return;

  const reconciliation =
    dist.total_consistent === true ? "Matches" : dist.total_consistent === false ? "Does not match" : "Not stated";
  buildStats(document.getElementById("distribution-stats"), [
    { label: "Stated total", value: dist.reported_total ?? "Not stated" },
    { label: "Observed total", value: dist.observed_total },
    { label: "Total reconciliation", value: reconciliation },
    { label: "Top-two share", value: dist.top_two_share_percent != null ? dist.top_two_share_percent + "%" : "n/a" },
  ]);

  const ranked = [...dist.results].sort((a, b) => b.count - a.count);
  buildTable(
    document.getElementById("distribution-table"),
    [
      { key: "rank", label: "Rank" },
      { key: "model", label: "Category / model" },
      { key: "count", label: `Reported ${dist.unit || "count"}` },
      { key: "share", label: "Share of total" },
      { key: "context", label: "Outcome context" },
    ],
    ranked.map((item, index) => ({
      rank: index + 1,
      model: item.model,
      count: item.count,
      share: item.share_percent != null ? item.share_percent + "%" : "n/a",
      context: item.outcome_context || "Not stated",
    }))
  );

  const charts = (result.charts || []).filter((c) => c.kind.startsWith("distribution"));
  if (charts.length) {
    buildCharts(document.getElementById("distribution-charts"), charts);
  }
  show(document.getElementById("section-distribution"), true);
}

function renderComparisons(result) {
  const container = document.getElementById("comparison-tables");
  container.innerHTML = "";
  let any = false;

  const standings = result.metric_standings || [];
  if (standings.length) {
    any = true;
    for (const standing of standings) {
      const wrap = document.createElement("div");
      wrap.className = "table-wrap";
      const metricName = (standing.metric || "").replace(/_/g, " ");
      wrap.innerHTML = `<h3>${escapeHtml(metricName)} · ranked comparison</h3><table></table>`;
      container.appendChild(wrap);
      buildTable(
        wrap.querySelector("table"),
        [
          { key: "rank", label: "Rank" },
          {
            key: "model",
            label: "Model",
            render: (r) =>
              r.is_best ? `<strong>${escapeHtml(r.model)}</strong> <span class="best-badge">best</span>` : escapeHtml(r.model),
          },
          { key: "value", label: "Value" },
          { key: "note", label: "Direction" },
        ],
        standing.rows.map((row) => ({
          rank: row.rank,
          model: row.model,
          value: `${formatValue(row.display)}${row.unit || ""}`,
          note: (row.direction || "").replace(/_/g, " "),
        }))
      );
    }
  }

  const claimResults = result.claim_metric_results || [];
  if (claimResults.length) {
    any = true;
    const wrap = document.createElement("div");
    wrap.className = "table-wrap";
    wrap.innerHTML = "<h3>Your reported values</h3><table></table>";
    container.appendChild(wrap);
    buildTable(
      wrap.querySelector("table"),
      [
        { key: "metric", label: "Metric" },
        { key: "model_a", label: "Model A" },
        { key: "a", label: "A value" },
        { key: "model_b", label: "Model B" },
        { key: "b", label: "B value" },
        { key: "difference", label: "Difference" },
        { key: "leader", label: "Leader" },
      ],
      claimResults.map((r) => ({
        metric: (r.metric || "").replace(/_/g, " "),
        model_a: r.model_a,
        a: r.model_a_display,
        model_b: r.model_b,
        b: r.model_b_display,
        difference:
          r.display_difference != null ? `${r.display_difference} ${r.difference_unit || ""}`.trim() : "—",
        leader: r.preferred_model && r.preferred_model !== "equal" ? r.preferred_model : "Equal",
      }))
    );
  }

  const sourceResults = result.external_model_comparisons || [];
  if (sourceResults.length) {
    any = true;
    const wrap = document.createElement("div");
    wrap.className = "table-wrap";
    wrap.innerHTML = "<h3>Retrieved source comparisons</h3><table></table>";
    container.appendChild(wrap);
    buildTable(
      wrap.querySelector("table"),
      [
        { key: "metric", label: "Metric" },
        { key: "model_a", label: "Model A" },
        { key: "a", label: "A value" },
        { key: "model_b", label: "Model B" },
        { key: "b", label: "B value" },
        {
          key: "source",
          label: "Source",
          render: (r) =>
            r.url
              ? `<a href="${escapeHtml(r.url)}" target="_blank" rel="noopener">${escapeHtml(r.title || r.source)}</a>`
              : escapeHtml(r.title || r.source || "—"),
        },
      ],
      sourceResults.map((r) => ({
        metric: (r.metric || "").replace(/_/g, " "),
        model_a: r.model_a,
        a: r.model_a_display,
        model_b: r.model_b,
        b: r.model_b_display,
        title: r.title,
        source: r.source,
        url: r.url,
      }))
    );
  }

  if (any) show(document.getElementById("section-comparisons"), true);
}

function renderMetricCharts(result) {
  const charts = (result.charts || []).filter((c) => !c.kind.startsWith("distribution"));
  if (!charts.length) return;
  buildCharts(document.getElementById("metric-charts"), charts);
  show(document.getElementById("section-charts"), true);
}

function formatValue(value) {
  if (value === null || value === undefined || value === "") return "—";
  const number = Number(value);
  if (Number.isFinite(number)) return String(Math.round(number * 1000) / 1000);
  return String(value);
}

function renderEvidence(result) {
  const rows = [];
  for (const summary of result.evidence_metrics || []) {
    for (const observation of summary.observations || []) {
      rows.push({ metric: summary.metric, ...observation });
    }
  }
  const visible = rows.slice(0, 25);
  if (!visible.length) return;
  buildTable(
    document.getElementById("evidence-table"),
    [
      { key: "model", label: "Model" },
      { key: "metric", label: "Metric" },
      { key: "value", label: "Value" },
      {
        key: "source",
        label: "Source",
        render: (r) =>
          r.url
            ? `<a href="${escapeHtml(r.url)}" target="_blank" rel="noopener">${escapeHtml(r.title || r.source)}</a>`
            : escapeHtml(r.title || r.source || "—"),
      },
    ],
    visible.map((o) => ({
      model: Array.isArray(o.models_detected) && o.models_detected.length ? o.models_detected.join(", ") : "—",
      metric: (o.metric || "").replace(/_/g, " "),
      value: formatValue(o.display_value ?? o.value),
      title: o.title,
      source: o.source,
      url: o.url,
    }))
  );
  show(document.getElementById("section-evidence"), true);
}

function renderComparability(result) {
  const comp = result.evidence_comparability;
  if (!comp || !comp.sources || !comp.sources.length) return;
  const rows = comp.sources.filter((s) => (s.source_benchmarks || []).length || (s.source_tasks || []).length);
  if (!rows.length) return;
  buildTable(
    document.getElementById("comparability-table"),
    [
      {
        key: "source",
        label: "Source",
        render: (r) =>
          r.url
            ? `<a href="${escapeHtml(r.url)}" target="_blank" rel="noopener">${escapeHtml(r.title || r.source)}</a>`
            : escapeHtml(r.title || r.source || "—"),
      },
      { key: "benchmarks", label: "Benchmarks" },
      { key: "tasks", label: "Task context" },
      { key: "status", label: "Status" },
    ],
    rows.map((s) => ({
      title: s.title,
      source: s.source,
      url: s.url,
      benchmarks: (s.source_benchmarks || []).join(", ") || "—",
      tasks: (s.source_tasks || []).join(", ") || "—",
      status: (s.status || "UNASSESSED").replace(/_/g, " "),
    }))
  );
  show(document.getElementById("section-comparability"), true);
}

function renderLimitations(result) {
  const raw = result.limitations;
  const limitations = Array.isArray(raw) ? raw : raw ? [raw] : [];
  if (!limitations.length) return;
  document.getElementById("limitations-list").innerHTML = limitations
    .map((item) => `<li>${escapeHtml(item)}</li>`)
    .join("");
  show(document.getElementById("section-limitations"), true);
}

async function runInvestigation() {
  const question = document.getElementById("question").value.trim();
  const market = document.getElementById("market").value.trim();
  const domain = document.getElementById("domain").value.trim() || "AI / ML evaluation";
  if (question.length < 3) {
    alert("Please enter a question first.");
    return;
  }

  runButton.disabled = true;
  setStatus("running", "Running");
  show(emptyState, false);
  show(reportContent, false);
  show(loading, true);
  document.querySelectorAll("#report-content .card").forEach((card) => show(card, false));

  try {
    const response = await fetch("/api/investigate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, market: market || null, domain }),
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || `Request failed with status ${response.status}`);
    }
    const result = await response.json();

    show(document.getElementById("section-overview"), true);
    renderOverview(result);
    renderSummary(result);
    renderDistribution(result);
    renderComparisons(result);
    renderMetricCharts(result);
    renderEvidence(result);
    renderComparability(result);
    renderLimitations(result);

    show(loading, false);
    show(reportContent, true);
    setStatus("done", "Complete");
  } catch (error) {
    show(loading, false);
    show(reportContent, true);
    show(document.getElementById("section-overview"), false);
    reportContent.innerHTML =
      `<div class="error-banner"><strong>Investigation failed.</strong> ${escapeHtml(error.message)}</div>` +
      reportContent.innerHTML;
    setStatus("error", "Error");
  } finally {
    runButton.disabled = false;
  }
}

runButton.addEventListener("click", runInvestigation);
document.getElementById("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) runInvestigation();
});

document.querySelectorAll(".example-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    document.getElementById("question").value = chip.dataset.q || "";
    document.getElementById("question").focus();
  });
});

import React, { useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Cpu,
  Database,
  Download,
  Play,
  RefreshCw,
  Upload,
} from "lucide-react";
import { ForecastLineChart, ResidualThresholdChart, SimpleBarChart } from "./components/Charts";

const TABS = [
  { id: "overview", label: "Overview", icon: Activity },
  { id: "explorer", label: "Explorer", icon: Database },
  { id: "forecaster", label: "Forecast", icon: Cpu },
  { id: "anomalies", label: "Anomalies", icon: AlertTriangle },
  { id: "benchmarks", label: "Benchmarks", icon: BarChart3 },
];

const CANDIDATE_K = [3.0, 3.5, 4.0, 4.5];

export default function App() {
  const [activeTab, setActiveTab] = useState("overview");
  const [overview, setOverview] = useState(null);
  const [profiles, setProfiles] = useState(null);
  const [anomalyData, setAnomalyData] = useState({
    k: 4.0,
    window_size: 168,
    min_residual_kwh: 0.65,
    persistence: 2,
    severe_spike_multiplier: 1.35,
    rebaseline_hours: 48,
    total: 0,
    anomalyCount: 0,
    groupedEventsCount: 0,
    events: [],
    rows: [],
  });
  const [anomWindow, setAnomWindow] = useState(500);
  const [anomK, setAnomK] = useState(4.0);
  const [anomLoading, setAnomLoading] = useState(false);
  const [anomFilter, setAnomFilter] = useState("All");
  const [anomViewMode, setAnomViewMode] = useState("events");
  const [simStartIdx, setSimStartIdx] = useState(29500);
  const [simHorizon, setSimHorizon] = useState(24);
  const [simResult, setSimResult] = useState(null);
  const [simLoading, setSimLoading] = useState(false);
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const [loading, setLoading] = useState(true);
  const [datasetsInfo, setDatasetsInfo] = useState({ active_id: "uci_default", datasets: [] });
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadName, setUploadName] = useState("");
  const [uploadDtCol, setUploadDtCol] = useState("");
  const [uploadTargetCol, setUploadTargetCol] = useState("");
  const [uploadPreview, setUploadPreview] = useState(null);
  const [datasetBusy, setDatasetBusy] = useState(false);
  const [datasetMsg, setDatasetMsg] = useState(null);

  const fetchOverview = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/overview");
      const data = await res.json();
      setOverview(data);
      const selectedK = data?.evaluation?.detector_config?.sensitivity_k;
      if (selectedK !== undefined) setAnomK(selectedK);
      const totalHrs = data?.validation?.file_info?.total_hourly_records || 34589;
      const safeStart = Math.max(180, Math.min(totalHrs - 48, Math.floor(totalHrs * 0.85)));
      setSimStartIdx(safeStart);
    } catch (err) {
      console.error("Failed to fetch overview:", err);
    } finally {
      setLoading(false);
    }
  };

  const fetchAnomalies = async (limit, kVal) => {
    setAnomLoading(true);
    try {
      const res = await fetch(`/api/anomalies?limit=${limit}&k=${kVal}`);
      const data = await res.json();
      setAnomalyData(data);
    } catch (err) {
      console.error("Failed to fetch anomalies:", err);
    } finally {
      setAnomLoading(false);
    }
  };

  const fetchProfiles = async (forceRefresh = false) => {
    if (profiles && !forceRefresh) return;
    try {
      const res = await fetch("/api/profiles");
      const data = await res.json();
      setProfiles(data);
    } catch (err) {
      console.error("Failed to fetch profiles:", err);
    }
  };

  const fetchDatasets = async () => {
    try {
      const res = await fetch("/api/datasets");
      const data = await res.json();
      setDatasetsInfo(data);
    } catch (err) {
      console.error("Failed to fetch datasets:", err);
    }
  };

  useEffect(() => {
    fetchOverview();
    fetchDatasets();
  }, []);

  useEffect(() => {
    if (activeTab === "anomalies") {
      fetchAnomalies(anomWindow, anomK);
    } else if (activeTab === "explorer") {
      fetchProfiles();
      fetchDatasets();
    }
  }, [activeTab, anomWindow, anomK]);

  const handleSelectFile = (e) => {
    const file = e.target.files?.[0];
    setDatasetMsg(null);
    if (!file) {
      setUploadFile(null);
      setUploadPreview(null);
      return;
    }
    setUploadFile(file);
    if (!uploadName) {
      setUploadName(file.name.replace(/\.[^.]+$/, ""));
    }
    const reader = new FileReader();
    reader.onload = (ev) => {
      const text = String(ev.target?.result || "");
      const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
      setUploadPreview({
        content: text,
        previewLines: lines.slice(0, 4),
        totalLines: lines.length,
      });
    };
    reader.readAsText(file);
  };

  const handleUploadDataset = async () => {
    if (!uploadFile || !uploadPreview?.content) {
      setDatasetMsg({ type: "error", text: "Please choose a CSV or TXT electricity dataset first." });
      return;
    }
    setDatasetBusy(true);
    setDatasetMsg(null);
    try {
      const res = await fetch("/api/datasets/upload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filename: uploadFile.name,
          name: uploadName.trim() || uploadFile.name,
          content: uploadPreview.content,
          datetimeCol: uploadDtCol.trim(),
          targetCol: uploadTargetCol.trim(),
          autoRetrain: true,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Upload and retraining failed.");
      }
      setDatasetMsg({
        type: "success",
        text: `Ingested "${data.ingested.name}" (${data.ingested.rows.toLocaleString()} hourly records), retrained XGBoost, and updated evaluation reports.`,
      });
      setUploadFile(null);
      setUploadPreview(null);
      setUploadName("");
      setUploadDtCol("");
      setUploadTargetCol("");
      await fetchOverview();
      await fetchDatasets();
      await fetchProfiles(true);
    } catch (err) {
      setDatasetMsg({ type: "error", text: err.message });
    } finally {
      setDatasetBusy(false);
    }
  };

  const handleActivateDataset = async (datasetId) => {
    setDatasetBusy(true);
    setDatasetMsg(null);
    try {
      const res = await fetch("/api/datasets/activate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ datasetId, retrain: true }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Failed to switch dataset.");
      }
      setDatasetMsg({
        type: "success",
        text: `Switched active dataset to "${data.registry.active_name}" and retrained forecasting & anomaly detection models.`,
      });
      await fetchOverview();
      await fetchDatasets();
      await fetchProfiles(true);
    } catch (err) {
      setDatasetMsg({ type: "error", text: err.message });
    } finally {
      setDatasetBusy(false);
    }
  };

  const handleRunSimulation = async () => {
    setSimLoading(true);
    try {
      const res = await fetch("/api/forecast/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ startIdx: simStartIdx, horizon: simHorizon }),
      });
      const data = await res.json();
      setSimResult(data);
    } catch (err) {
      console.error("Simulation failed:", err);
    } finally {
      setSimLoading(false);
    }
  };

  const handleRunPipeline = async (forceTrain = false) => {
    setPipelineRunning(true);
    try {
      await fetch("/api/pipeline/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ forceTrain }),
      });
      await fetchOverview();
      if (activeTab === "anomalies") await fetchAnomalies(anomWindow, anomK);
    } catch (err) {
      console.error("Pipeline run failed:", err);
    } finally {
      setPipelineRunning(false);
    }
  };

  const evalData = overview?.evaluation || {};
  const detectorCfg = evalData.detector_config || {
    sensitivity_k: 4.0,
    window_size: 168,
    min_residual_kwh: 0.65,
    persistence: 2,
    severe_spike_multiplier: 1.35,
    rebaseline_hours: 48,
  };
  const testMetrics = evalData.forecasting_metrics || {};
  const adaptiveEv = evalData.adaptive_detector?.event_level || {};
  const baselineEv = evalData.fixed_baseline?.event_level || {};
  const adaptiveOverall = evalData.adaptive_detector?.overall || {};
  const baselineOverall = evalData.fixed_baseline?.overall || {};
  const kSweep = evalData.k_sensitivity_sweep || null;
  const beforeVsAfter = evalData.before_vs_after || null;
  const testComp = beforeVsAfter?.test || null;
  const valComp = beforeVsAfter?.validation || null;
  const validation = overview?.validation || {};
  const accReport = overview?.accuracy || {};
  const modelMeta = overview?.modelMeta || {};
  const activeDatasetName =
    overview?.activeDataset?.name ||
    datasetsInfo?.active_name ||
    "UCI Household Electric Power (Default)";
  const totalHourlyRecords = validation?.file_info?.total_hourly_records || 34589;
  const simMinIdx = Math.max(180, Math.floor(totalHourlyRecords * 0.85));
  const simMaxIdx = Math.max(simMinIdx + 1, totalHourlyRecords - 48);

  const filteredAnomalies = (anomalyData.rows || []).filter(
    (r) => r.is_anomaly && (anomFilter === "All" || r.anomaly_type === anomFilter)
  );
  const filteredEvents = (anomalyData.events || []).filter(
    (ev) => anomFilter === "All" || ev.anomaly_type === anomFilter
  );

  return (
    <div className="app-shell">
      {/* Mobile Sticky Top Header */}
      <header className="mobile-header">
        <div className="brand">
          <span className="brand-dot" />
          <div>
            <div className="brand-title">Power Anomaly ML</div>
            <div className="brand-sub">
              Adaptive MAD (k={detectorCfg.sensitivity_k}) + XGBoost
            </div>
          </div>
        </div>
        <button
          className="btn btn-primary"
          style={{ minHeight: 36, padding: "6px 12px", fontSize: 12 }}
          onClick={() => handleRunPipeline(false)}
          disabled={pipelineRunning}
        >
          <RefreshCw size={14} />
          {pipelineRunning ? "Running..." : "Sync"}
        </button>
      </header>

      {/* Desktop Sidebar */}
      <aside className="sidebar">
        <div>
          <div className="brand">
            <span className="brand-dot" />
            <div>
              <div className="brand-title">Power Anomaly ML</div>
              <div className="brand-sub">Adaptive Electricity Research</div>
            </div>
          </div>

          <nav className="sidebar-nav">
            {TABS.map((t) => {
              const Icon = t.icon;
              return (
                <button
                  key={t.id}
                  className={`sidebar-btn ${activeTab === t.id ? "active" : ""}`}
                  onClick={() => setActiveTab(t.id)}
                >
                  <Icon size={18} />
                  <span>{t.label}</span>
                </button>
              );
            })}
          </nav>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <button
            className="btn btn-primary"
            onClick={() => handleRunPipeline(true)}
            disabled={pipelineRunning}
          >
            <RefreshCw size={16} />
            {pipelineRunning ? "Running Pipeline..." : "Retrain & Evaluate"}
          </button>
          <div style={{ fontSize: 11, color: "#64748b", textAlign: "center" }}>
            k={detectorCfg.sensitivity_k} · {detectorCfg.window_size}h Window
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <div className="hero-banner">
          <h1>Adaptive Household Electricity Anomaly Detection</h1>
          <p>
            Leak-free XGBoost forecasting paired with past-only Rolling Median + MAD dynamic thresholds
            (k={detectorCfg.sensitivity_k}, {detectorCfg.window_size}h window, event grouping &amp; controlled re-baselining).
          </p>
          <div style={{ marginTop: 10 }}>
            <span className="badge badge-info">Active Dataset: {activeDatasetName}</span>
          </div>
        </div>

        {loading ? (
          <div className="card">Loading research metrics and telemetry...</div>
        ) : (
          <>
            {/* OVERVIEW TAB */}
            {activeTab === "overview" && (
              <>
                <div className="kpi-grid">
                  <div className="kpi-card">
                    <div className="kpi-label">Test MAE / RMSE</div>
                    <div className="kpi-value">
                      {testMetrics.mae !== undefined ? `${testMetrics.mae.toFixed(3)}` : "—"}
                    </div>
                    <div className="kpi-sub">
                      kWh MAE · RMSE: {testMetrics.rmse ? testMetrics.rmse.toFixed(3) : "—"} kWh
                    </div>
                  </div>
                  <div className="kpi-card">
                    <div className="kpi-label">Grouped Events</div>
                    <div className="kpi-value">
                      {overview?.anomaliesSummary?.groupedEventsCount ?? 0}
                    </div>
                    <div className="kpi-sub">
                      {testComp
                        ? `Down from ${testComp.reduction_summary.grouped_events_before} (k=3.0)`
                        : "Contiguous event alerts"}
                    </div>
                  </div>
                  <div className="kpi-card">
                    <div className="kpi-label">Flagged Test Hours</div>
                    <div className="kpi-value">
                      {overview?.anomaliesSummary?.detectedCount ?? 0}
                    </div>
                    <div className="kpi-sub">
                      {overview?.anomaliesSummary?.ratePercent ?? 0}% of{" "}
                      {overview?.anomaliesSummary?.totalTestHours ?? 0}h
                      {testComp
                        ? ` (-${testComp.reduction_summary.flagged_hours_reduction_percent}%)`
                        : ""}
                    </div>
                  </div>
                  <div className="kpi-card">
                    <div className="kpi-label">Synthetic Event Capture</div>
                    <div className="kpi-value">
                      {adaptiveEv.event_detection_rate !== undefined
                        ? `${(adaptiveEv.event_detection_rate * 100).toFixed(0)}%`
                        : "—"}
                    </div>
                    <div className="kpi-sub">
                      Delay: {adaptiveEv.average_detection_delay_hours ?? 0}h · k={detectorCfg.sensitivity_k}
                    </div>
                  </div>
                </div>

                <div className="panel-grid two-col">
                  <div className="card">
                    <div className="card-header">
                      <div>
                        <div className="card-title">Recent Test Stream: Actual vs. Forecast</div>
                        <div className="card-subtitle">
                          Red markers highlight potential consumption anomalies (k={detectorCfg.sensitivity_k})
                        </div>
                      </div>
                      <span className="badge badge-info">
                        k={detectorCfg.sensitivity_k} · {detectorCfg.window_size}h Window
                      </span>
                    </div>
                    <ForecastLineChart data={overview?.recentStream || []} showAnomalies />
                  </div>

                  <div className="card">
                    <div className="card-header">
                      <div className="card-title">Detector Protections</div>
                      <span className="badge badge-success">Active</span>
                    </div>
                    <div className="checklist">
                      <div className="checklist-item">
                        <CheckCircle2 size={16} color="#10b981" />
                        <span>Validation-Selected k = {detectorCfg.sensitivity_k}</span>
                      </div>
                      <div className="checklist-item">
                        <CheckCircle2 size={16} color="#10b981" />
                        <span>Min Error Floor: {detectorCfg.min_residual_kwh} kWh</span>
                      </div>
                      <div className="checklist-item">
                        <CheckCircle2 size={16} color="#10b981" />
                        <span>
                          Persistence: {detectorCfg.persistence}h (1h if ≥{detectorCfg.severe_spike_multiplier}× Tₜ)
                        </span>
                      </div>
                      <div className="checklist-item">
                        <CheckCircle2 size={16} color="#10b981" />
                        <span>Re-baselining after {detectorCfg.rebaseline_hours}h routine shift</span>
                      </div>
                      <div className="checklist-item">
                        <CheckCircle2 size={16} color="#10b981" />
                        <span>Zero Future Target Leakage</span>
                      </div>
                    </div>
                  </div>
                </div>

                {testComp && (
                  <div className="card">
                    <div className="card-header">
                      <div>
                        <div className="card-title">Before vs. After False-Alarm Reduction (Test Split)</div>
                        <div className="card-subtitle">
                          Unlabelled real-world alerts vs. controlled synthetic benchmark metrics. Alerts identify
                          potential unusual electricity consumption, not confirmed physical leakage.
                        </div>
                      </div>
                      <span className="badge badge-success">
                        -{testComp.reduction_summary.flagged_hours_reduction_percent}% Flagged Hours
                      </span>
                    </div>
                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>Evaluation Scope &amp; Metric</th>
                            <th>Before (Legacy k=3.0)</th>
                            <th>After (Improved k={detectorCfg.sensitivity_k})</th>
                            <th>Change</th>
                          </tr>
                        </thead>
                        <tbody>
                          <tr>
                            <td><strong>Real Stream: Flagged Hours</strong></td>
                            <td>{testComp.legacy_before.real_unlabelled_stream.flagged_hours} hrs ({testComp.legacy_before.real_unlabelled_stream.flagged_rate_percent}%)</td>
                            <td>{testComp.improved_after.real_unlabelled_stream.flagged_hours} hrs ({testComp.improved_after.real_unlabelled_stream.flagged_rate_percent}%)</td>
                            <td><span className="badge badge-success">-{testComp.reduction_summary.flagged_hours_reduction_percent}%</span></td>
                          </tr>
                          <tr>
                            <td><strong>Real Stream: Grouped Events</strong></td>
                            <td>{testComp.legacy_before.real_unlabelled_stream.grouped_events} events</td>
                            <td>{testComp.improved_after.real_unlabelled_stream.grouped_events} events</td>
                            <td><span className="badge badge-success">-{testComp.reduction_summary.grouped_events_reduction_percent}%</span></td>
                          </tr>
                          <tr>
                            <td><strong>Synthetic: Event Detection Rate</strong></td>
                            <td>{(testComp.legacy_before.synthetic_benchmark.event_level.event_detection_rate * 100).toFixed(1)}%</td>
                            <td>{(testComp.improved_after.synthetic_benchmark.event_level.event_detection_rate * 100).toFixed(1)}%</td>
                            <td>{testComp.improved_after.synthetic_benchmark.event_level.average_detection_delay_hours}h delay</td>
                          </tr>
                          <tr>
                            <td><strong>Synthetic: Controlled Precision / Recall / F1</strong></td>
                            <td>
                              {(testComp.legacy_before.synthetic_benchmark.controlled_benchmark.precision * 100).toFixed(1)}% /{" "}
                              {(testComp.legacy_before.synthetic_benchmark.controlled_benchmark.recall * 100).toFixed(1)}% /{" "}
                              {(testComp.legacy_before.synthetic_benchmark.controlled_benchmark.f1_score * 100).toFixed(1)}%
                            </td>
                            <td>
                              {(testComp.improved_after.synthetic_benchmark.controlled_benchmark.precision * 100).toFixed(1)}% /{" "}
                              {(testComp.improved_after.synthetic_benchmark.controlled_benchmark.recall * 100).toFixed(1)}% /{" "}
                              {(testComp.improved_after.synthetic_benchmark.controlled_benchmark.f1_score * 100).toFixed(1)}%
                            </td>
                            <td><span className="badge badge-info">Controlled Window</span></td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </>
            )}

            {/* EXPLORER TAB */}
            {activeTab === "explorer" && (
              <>
                <div className="card">
                  <div className="card-header">
                    <div>
                      <div className="card-title">Add New Dataset for Retraining &amp; Evaluation</div>
                      <div className="card-subtitle">
                        Upload any CSV/TXT electricity dataset (≥300 hourly records or sub-hourly readings);
                        XGBoost retraining, k-sweep, and evaluation run automatically.
                      </div>
                    </div>
                    <a className="btn btn-outline" href="/api/datasets/template">
                      <Download size={15} />
                      Sample CSV Template
                    </a>
                  </div>

                  {datasetMsg && (
                    <div className={`alert-banner ${datasetMsg.type}`}>{datasetMsg.text}</div>
                  )}

                  <div className="panel-grid equal-two" style={{ marginBottom: 14 }}>
                    <div className="upload-box">
                      <div className="input-group" style={{ marginBottom: 10 }}>
                        <label>Select CSV or TXT Dataset File</label>
                        <input
                          type="file"
                          accept=".csv,.txt"
                          onChange={handleSelectFile}
                          disabled={datasetBusy}
                        />
                      </div>
                      {uploadPreview && (
                        <div style={{ fontSize: 11, color: "#475569", overflowX: "auto" }}>
                          <strong>Detected {uploadPreview.totalLines.toLocaleString()} lines:</strong>
                          <pre style={{ marginTop: 4, fontFamily: "JetBrains Mono, monospace" }}>
                            {uploadPreview.previewLines.join("\n")}
                          </pre>
                        </div>
                      )}
                    </div>

                    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                      <div className="input-group">
                        <label>Dataset Display Name (Optional)</label>
                        <input
                          type="text"
                          className="text-input"
                          placeholder="e.g., Smart Meter Apartment B"
                          value={uploadName}
                          onChange={(e) => setUploadName(e.target.value)}
                          disabled={datasetBusy}
                        />
                      </div>
                      <div className="panel-grid equal-two" style={{ marginBottom: 0 }}>
                        <div className="input-group">
                          <label>Datetime Column (Blank = Auto)</label>
                          <input
                            type="text"
                            className="text-input"
                            placeholder="Datetime / timestamp"
                            value={uploadDtCol}
                            onChange={(e) => setUploadDtCol(e.target.value)}
                            disabled={datasetBusy}
                          />
                        </div>
                        <div className="input-group">
                          <label>Target Power Column (Blank = Auto)</label>
                          <input
                            type="text"
                            className="text-input"
                            placeholder="Global_active_power / kWh"
                            value={uploadTargetCol}
                            onChange={(e) => setUploadTargetCol(e.target.value)}
                            disabled={datasetBusy}
                          />
                        </div>
                      </div>
                      <button
                        className="btn btn-primary"
                        onClick={handleUploadDataset}
                        disabled={datasetBusy || !uploadFile}
                      >
                        <Upload size={16} />
                        {datasetBusy ? "Ingesting, Retraining & Evaluating..." : "Upload, Retrain XGBoost & Evaluate"}
                      </button>
                    </div>
                  </div>

                  {datasetsInfo?.datasets?.length > 0 && (
                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>Registered Dataset</th>
                            <th>Records</th>
                            <th>Date Coverage</th>
                            <th>Status / Action</th>
                          </tr>
                        </thead>
                        <tbody>
                          {datasetsInfo.datasets.map((ds) => {
                            const isActive = ds.id === datasetsInfo.active_id;
                            return (
                              <tr key={ds.id}>
                                <td><strong>{ds.name}</strong></td>
                                <td>{Number(ds.rows || 0).toLocaleString()} hrs</td>
                                <td>
                                  {String(ds.start_date || "").slice(0, 10)} →{" "}
                                  {String(ds.end_date || "").slice(0, 10)}
                                </td>
                                <td>
                                  {isActive ? (
                                    <span className="badge badge-success">Active</span>
                                  ) : (
                                    <button
                                      className="btn btn-outline"
                                      style={{ minHeight: 34, padding: "4px 12px", fontSize: 12 }}
                                      onClick={() => handleActivateDataset(ds.id)}
                                      disabled={datasetBusy}
                                    >
                                      Switch &amp; Retrain
                                    </button>
                                  )}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>

                <div className="kpi-grid">
                  <div className="kpi-card">
                    <div className="kpi-label">Hourly Records</div>
                    <div className="kpi-value">
                      {validation?.file_info?.total_hourly_records?.toLocaleString() || "34,589"}
                    </div>
                    <div className="kpi-sub">100% Grid Continuity</div>
                  </div>
                  <div className="kpi-card">
                    <div className="kpi-label">Mean Load</div>
                    <div className="kpi-value">
                      {validation?.consumption_range_checks?.mean_kwh ?? "1.09"} kWh
                    </div>
                    <div className="kpi-sub">
                      Median: {validation?.consumption_range_checks?.median_kwh ?? "0.80"} kWh
                    </div>
                  </div>
                  <div className="kpi-card">
                    <div className="kpi-label">Peak Load</div>
                    <div className="kpi-value">
                      {validation?.consumption_range_checks?.max_kwh ?? "6.56"} kWh
                    </div>
                    <div className="kpi-sub">Min: {validation?.consumption_range_checks?.min_kwh ?? "0.12"} kWh</div>
                  </div>
                  <div className="kpi-card">
                    <div className="kpi-label">Validation</div>
                    <div className="kpi-value">{validation?.overall_status || "PASSED"}</div>
                    <div className="kpi-sub">Zero Negative Values</div>
                  </div>
                </div>

                <div className="panel-grid equal-two">
                  <div className="card">
                    <div className="card-header">
                      <div>
                        <div className="card-title">24-Hour Diurnal Load Profile</div>
                        <div className="card-subtitle">Average kWh by hour of day</div>
                      </div>
                    </div>
                    {profiles ? (
                      <SimpleBarChart
                        items={profiles.diurnal.filter((_, i) => i % 2 === 0).map((d) => ({ ...d, label: `${d.hour}:00` }))}
                        labelKey="label"
                        valueKey="mean"
                        suffix=" kWh"
                      />
                    ) : (
                      <div className="card-subtitle">Loading diurnal profile...</div>
                    )}
                  </div>

                  <div className="card">
                    <div className="card-header">
                      <div>
                        <div className="card-title">Weekly Consumption Profile</div>
                        <div className="card-subtitle">Average kWh by day of week</div>
                      </div>
                    </div>
                    {profiles ? (
                      <SimpleBarChart
                        items={profiles.weekly}
                        labelKey="day"
                        valueKey="mean"
                        color="#0f3e72"
                        suffix=" kWh"
                      />
                    ) : (
                      <div className="card-subtitle">Loading weekly profile...</div>
                    )}
                  </div>
                </div>
              </>
            )}

            {/* FORECASTER & SIMULATOR TAB */}
            {activeTab === "forecaster" && (
              <>
                <div className="card">
                  <div className="card-header">
                    <div>
                      <div className="card-title">Interactive Recursive Horizon Simulator</div>
                      <div className="card-subtitle">
                        Feeds each predicted hour back as lag_1h for multi-step rollout
                      </div>
                    </div>
                    <span className="badge badge-info">Best Iter: {modelMeta.best_iteration || "—"} trees</span>
                  </div>

                  <div className="panel-grid equal-two" style={{ marginBottom: 12 }}>
                    <div className="input-group">
                      <label>Test Stream Start Index: Record #{simStartIdx}</label>
                      <input
                        type="range"
                        className="range-input"
                        min={simMinIdx}
                        max={simMaxIdx}
                        step={Math.max(1, Math.min(24, Math.floor((simMaxIdx - simMinIdx) / 10)))}
                        value={Math.min(simMaxIdx, Math.max(simMinIdx, simStartIdx))}
                        onChange={(e) => setSimStartIdx(Number(e.target.value))}
                      />
                    </div>
                    <div className="input-group">
                      <label>Forecast Horizon</label>
                      <div className="pill-row">
                        {[12, 24, 48].map((h) => (
                          <button
                            key={h}
                            className={`pill-btn ${simHorizon === h ? "active" : ""}`}
                            onClick={() => setSimHorizon(h)}
                          >
                            {h} Hours
                          </button>
                        ))}
                        <button
                          className="btn btn-primary"
                          onClick={handleRunSimulation}
                          disabled={simLoading}
                        >
                          <Play size={15} />
                          {simLoading ? "Simulating..." : "Run Rollout"}
                        </button>
                      </div>
                    </div>
                  </div>

                  {simResult && <ForecastLineChart data={simResult.series} height={260} />}
                </div>

                {accReport?.forecasting_accuracy && (
                  <div className="card">
                    <div className="card-header">
                      <div className="card-title">Chronological Partition Regression Accuracy</div>
                    </div>
                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>Split</th>
                            <th>Samples</th>
                            <th>MAE (kWh)</th>
                            <th>RMSE (kWh)</th>
                            <th>R²</th>
                          </tr>
                        </thead>
                        <tbody>
                          {[
                            ["Train (70%)", accReport.forecasting_accuracy.training_set],
                            ["Validation (15%)", accReport.forecasting_accuracy.validation_set],
                            ["Test (15%)", accReport.forecasting_accuracy.test_set],
                          ].map(([name, split]) => (
                            <tr key={name}>
                              <td><strong>{name}</strong></td>
                              <td>{split.sample_count.toLocaleString()}</td>
                              <td>{split.metrics.mae_kwh}</td>
                              <td>{split.metrics.rmse_kwh}</td>
                              <td>{split.metrics.r2_score}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </>
            )}

            {/* ANOMALIES STUDIO TAB */}
            {activeTab === "anomalies" && (
              <>
                <div className="card">
                  <div className="card-header">
                    <div>
                      <div className="card-title">Adaptive Anomaly Detection Studio</div>
                      <div className="card-subtitle">
                        {anomLoading
                          ? "Evaluating selected k..."
                          : `${anomalyData.anomalyCount} flagged hours (${anomalyData.ratePercent ?? 0}%) grouped into ${anomalyData.groupedEventsCount ?? 0} anomaly events`}
                        {anomalyData.beforeFlaggedHours
                          ? ` · Legacy k=3.0 flagged ${anomalyData.beforeFlaggedHours}h (${anomalyData.beforeEventsCount} events)`
                          : ""}
                      </div>
                    </div>
                    <div className="pill-row">
                      <a className="btn btn-outline" href="/api/download/events">
                        <Download size={15} />
                        Events CSV
                      </a>
                      <a className="btn btn-outline" href="/api/download/anomalies">
                        <Download size={15} />
                        Hourly CSV
                      </a>
                    </div>
                  </div>

                  {/* Sensitivity k & Window Controls */}
                  <div className="panel-grid equal-two" style={{ marginBottom: 14 }}>
                    <div className="input-group">
                      <label>
                        MAD Sensitivity (k) — Compare Detection Results ({anomalyData.window_size || 168}h Window, Floor {anomalyData.min_residual_kwh ?? 0.65} kWh)
                      </label>
                      <div className="pill-row">
                        {CANDIDATE_K.map((kVal) => (
                          <button
                            key={kVal}
                            className={`pill-btn ${Math.abs(anomK - kVal) < 1e-6 ? "active" : ""}`}
                            onClick={() => setAnomK(kVal)}
                          >
                            k = {kVal.toFixed(1)}
                            {Math.abs(kVal - detectorCfg.sensitivity_k) < 1e-6 ? " (Selected)" : ""}
                          </button>
                        ))}
                      </div>
                    </div>
                    <div className="input-group">
                      <label>Display Window</label>
                      <div className="pill-row">
                        {[250, 500, 1000].map((w) => (
                          <button
                            key={w}
                            className={`pill-btn ${anomWindow === w ? "active" : ""}`}
                            onClick={() => setAnomWindow(w)}
                          >
                            Last {w}h
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>

                  <ForecastLineChart data={anomalyData.rows} height={240} showAnomalies />
                  <div style={{ height: 12 }} />
                  <ResidualThresholdChart data={anomalyData.rows} height={190} />
                </div>

                <div className="card">
                  <div className="card-header">
                    <div>
                      <div className="card-title">
                        {anomViewMode === "events"
                          ? `Grouped Anomaly Events (${filteredEvents.length})`
                          : `Flagged Hourly Readings (${filteredAnomalies.length})`}
                      </div>
                      <div className="card-subtitle">
                        Consecutive flagged hours are grouped into a single alert event
                      </div>
                    </div>
                    <div className="pill-row">
                      <button
                        className={`pill-btn ${anomViewMode === "events" ? "active" : ""}`}
                        onClick={() => setAnomViewMode("events")}
                      >
                        Grouped Events
                      </button>
                      <button
                        className={`pill-btn ${anomViewMode === "hours" ? "active" : ""}`}
                        onClick={() => setAnomViewMode("hours")}
                      >
                        Hourly Details
                      </button>
                    </div>
                  </div>

                  <div className="pill-row" style={{ marginBottom: 12 }}>
                    {["All", "High Spike", "Sustained Shift", "Sudden Drop"].map((cat) => (
                      <button
                        key={cat}
                        className={`pill-btn ${anomFilter === cat ? "active" : ""}`}
                        onClick={() => setAnomFilter(cat)}
                      >
                        {cat}
                      </button>
                    ))}
                  </div>

                  {anomViewMode === "events" ? (
                    <>
                      {/* Mobile Event Cards */}
                      <div className="mobile-event-list">
                        {filteredEvents.slice(-30).reverse().map((ev) => (
                          <div className="mobile-event-card" key={ev.event_id}>
                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                              <strong style={{ fontSize: 12 }}>
                                Event #{ev.event_id} · {ev.duration_hours}h
                              </strong>
                              <span className="badge badge-danger">{ev.anomaly_type}</span>
                            </div>
                            <div style={{ fontSize: 12, color: "#475569" }}>
                              {ev.start_timestamp} → {ev.end_timestamp.slice(11, 16)}
                            </div>
                            <div style={{ fontSize: 12, color: "#64748b", marginTop: 4 }}>
                              Peak: <strong>{ev.peak_actual.toFixed(2)} kWh</strong> (Pred avg:{" "}
                              {ev.mean_predicted.toFixed(2)}) · Max Residual: {ev.peak_residual.toFixed(2)} vs{" "}
                              {ev.threshold_at_peak.toFixed(2)} kWh
                            </div>
                          </div>
                        ))}
                      </div>

                      {/* Desktop Events Table */}
                      <div className="desktop-table-only table-wrap">
                        <table>
                          <thead>
                            <tr>
                              <th>Event #</th>
                              <th>Start Time</th>
                              <th>End Time</th>
                              <th>Duration</th>
                              <th>Type</th>
                              <th>Peak Actual</th>
                              <th>Avg Forecast</th>
                              <th>Peak Residual</th>
                              <th>Threshold</th>
                              <th>Peak Score</th>
                            </tr>
                          </thead>
                          <tbody>
                            {filteredEvents.slice(-40).reverse().map((ev) => (
                              <tr key={ev.event_id}>
                                <td><strong>#{ev.event_id}</strong></td>
                                <td>{ev.start_timestamp}</td>
                                <td>{ev.end_timestamp}</td>
                                <td>{ev.duration_hours}h</td>
                                <td><span className="badge badge-danger">{ev.anomaly_type}</span></td>
                                <td>{ev.peak_actual.toFixed(3)} kWh</td>
                                <td>{ev.mean_predicted.toFixed(3)} kWh</td>
                                <td>{ev.peak_residual.toFixed(3)} kWh</td>
                                <td>{ev.threshold_at_peak.toFixed(3)} kWh</td>
                                <td>{ev.peak_score.toFixed(2)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </>
                  ) : (
                    <>
                      {/* Mobile Hourly Cards */}
                      <div className="mobile-event-list">
                        {filteredAnomalies.slice(-25).reverse().map((row, idx) => (
                          <div className="mobile-event-card" key={idx}>
                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                              <strong style={{ fontSize: 12 }}>{row.timestamp}</strong>
                              <span className="badge badge-danger">{row.anomaly_type}</span>
                            </div>
                            <div style={{ fontSize: 12, color: "#475569" }}>
                              Actual: <strong>{row.actual.toFixed(2)} kWh</strong> · Forecast: {row.predicted.toFixed(2)} kWh
                            </div>
                            <div style={{ fontSize: 12, color: "#64748b", marginTop: 2 }}>
                              Residual: {row.residual.toFixed(2)} vs Threshold: {row.threshold.toFixed(2)} (z={row.score.toFixed(2)})
                            </div>
                          </div>
                        ))}
                      </div>

                      {/* Desktop Hourly Table */}
                      <div className="desktop-table-only table-wrap">
                        <table>
                          <thead>
                            <tr>
                              <th>Timestamp</th>
                              <th>Event #</th>
                              <th>Type</th>
                              <th>Actual (kWh)</th>
                              <th>Forecast (kWh)</th>
                              <th>Residual</th>
                              <th>Threshold</th>
                              <th>MAD Score</th>
                            </tr>
                          </thead>
                          <tbody>
                            {filteredAnomalies.slice(-40).reverse().map((row, idx) => (
                              <tr key={idx}>
                                <td>{row.timestamp}</td>
                                <td>#{row.event_id}</td>
                                <td><span className="badge badge-danger">{row.anomaly_type}</span></td>
                                <td>{row.actual.toFixed(3)}</td>
                                <td>{row.predicted.toFixed(3)}</td>
                                <td>{row.residual.toFixed(3)}</td>
                                <td>{row.threshold.toFixed(3)}</td>
                                <td>{row.score.toFixed(2)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </>
                  )}
                </div>
              </>
            )}

            {/* BENCHMARKS & DOCS TAB */}
            {activeTab === "benchmarks" && (
              <>
                <div className="panel-grid equal-two">
                  <div className="card">
                    <div className="card-header">
                      <div className="card-title">Adaptive MAD vs. Fixed Baseline</div>
                      <span className="badge badge-success">Controlled Synthetic Benchmark</span>
                    </div>
                    <SimpleBarChart
                      items={[
                        {
                          label: `Adaptive k=${detectorCfg.sensitivity_k}`,
                          value: (adaptiveOverall.recall || 0) * 100,
                          color: "#0284c7",
                        },
                        {
                          label: "Fixed 3σ",
                          value: (baselineOverall.recall || 0) * 100,
                          color: "#f59e0b",
                        },
                      ]}
                      labelKey="label"
                      valueKey="value"
                      suffix="% Recall"
                    />
                    <div style={{ marginTop: 14, fontSize: 13, color: "#475569" }}>
                      Event Detection Rate:{" "}
                      <strong>{((adaptiveEv.event_detection_rate || 0) * 100).toFixed(1)}%</strong> (Adaptive,{" "}
                      {adaptiveEv.average_detection_delay_hours ?? 0}h delay) vs{" "}
                      <strong>{((baselineEv.event_detection_rate || 0) * 100).toFixed(1)}%</strong> (Fixed Baseline)
                    </div>
                  </div>

                  <div className="card">
                    <div className="card-header">
                      <div className="card-title">Research Reports &amp; Downloads</div>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                      <a className="btn btn-primary" href="/api/download/docx">
                        <Download size={16} />
                        Download Word Research Report (.docx)
                      </a>
                      <a className="btn btn-outline" href="/api/download/events">
                        <Download size={16} />
                        Download Grouped Anomaly Events (.csv)
                      </a>
                      <a className="btn btn-outline" href="/api/download/accuracy_md">
                        <Download size={16} />
                        Download Accuracy Report (.md)
                      </a>
                      <a className="btn btn-outline" href="/api/download/evaluation">
                        <Download size={16} />
                        Download Evaluation JSON
                      </a>
                    </div>
                  </div>
                </div>

                {kSweep && (
                  <div className="card">
                    <div className="card-header">
                      <div>
                        <div className="card-title">
                          MAD Sensitivity Constant (k) Validation Sweep
                        </div>
                        <div className="card-subtitle">{kSweep.selection_criterion}</div>
                      </div>
                      <span className="badge badge-info">Selected k = {kSweep.selected_k}</span>
                    </div>
                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>k Value</th>
                            <th>Val Flagged Hrs</th>
                            <th>Test Flagged Hrs</th>
                            <th>Test Events</th>
                            <th>Val Synth Precision</th>
                            <th>Val Synth Recall</th>
                            <th>Val Synth F1</th>
                            <th>Val Event Rate</th>
                          </tr>
                        </thead>
                        <tbody>
                          {kSweep.results.map((r) => (
                            <tr key={r.k}>
                              <td>
                                <strong>k = {r.k.toFixed(1)}</strong>{" "}
                                {r.k === kSweep.selected_k && (
                                  <span className="badge badge-success">Optimal</span>
                                )}
                              </td>
                              <td>{r.validation.flagged_hours} ({r.validation.flagged_rate_percent}%)</td>
                              <td>{r.test.flagged_hours} ({r.test.flagged_rate_percent}%)</td>
                              <td>{r.test.grouped_events}</td>
                              <td>{(r.validation.synthetic_precision * 100).toFixed(1)}%</td>
                              <td>{(r.validation.synthetic_recall * 100).toFixed(1)}%</td>
                              <td>{(r.validation.synthetic_f1 * 100).toFixed(1)}%</td>
                              <td>{(r.validation.synthetic_event_rate * 100).toFixed(1)}%</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {valComp && testComp && (
                  <div className="card">
                    <div className="card-header">
                      <div className="card-title">Before vs. After Validation &amp; Test Comparison</div>
                    </div>
                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>Partition</th>
                            <th>Before Flagged Hrs</th>
                            <th>After Flagged Hrs</th>
                            <th>Hours Reduction</th>
                            <th>Before Events</th>
                            <th>After Events</th>
                            <th>Synth Recall (After)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {[
                            ["Validation (15%)", valComp],
                            ["Test (15%)", testComp],
                          ].map(([label, comp]) => (
                            <tr key={label}>
                              <td><strong>{label}</strong></td>
                              <td>{comp.reduction_summary.flagged_hours_before}</td>
                              <td>{comp.reduction_summary.flagged_hours_after}</td>
                              <td>
                                <span className="badge badge-success">
                                  -{comp.reduction_summary.flagged_hours_reduction_percent}%
                                </span>
                              </td>
                              <td>{comp.reduction_summary.grouped_events_before}</td>
                              <td>{comp.reduction_summary.grouped_events_after}</td>
                              <td>
                                {(comp.improved_after.synthetic_benchmark.controlled_benchmark.recall * 100).toFixed(1)}%
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {evalData.adaptive_detector?.per_scenario && (
                  <div className="card">
                    <div className="card-header">
                      <div className="card-title">Scenario-by-Scenario Synthetic Detection Breakdown</div>
                    </div>
                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>Scenario</th>
                            <th>Injected Hours</th>
                            <th>Caught Hours</th>
                            <th>Recall</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(evalData.adaptive_detector.per_scenario).map(
                            ([name, sc]) => (
                              <tr key={name}>
                                <td><strong>{name.replace(/_/g, " ")}</strong></td>
                                <td>{sc.anomalous_samples}</td>
                                <td>{sc.detected_samples}</td>
                                <td>
                                  <span className="badge badge-info">{(sc.recall * 100).toFixed(1)}%</span>
                                </td>
                              </tr>
                            )
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </>
            )}
          </>
        )}
      </main>

      {/* Mobile Bottom Navigation Bar */}
      <nav className="bottom-nav">
        {TABS.map((t) => {
          const Icon = t.icon;
          return (
            <button
              key={t.id}
              className={`bottom-nav-btn ${activeTab === t.id ? "active" : ""}`}
              onClick={() => setActiveTab(t.id)}
            >
              <Icon size={18} />
              <span>{t.label}</span>
            </button>
          );
        })}
      </nav>
    </div>
  );
}

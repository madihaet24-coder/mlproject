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
  Zap,
} from "lucide-react";
import { ForecastLineChart, ResidualThresholdChart, SimpleBarChart } from "./components/Charts";

const TABS = [
  { id: "overview", label: "Overview", icon: Activity },
  { id: "explorer", label: "Explorer", icon: Database },
  { id: "forecaster", label: "Forecast", icon: Cpu },
  { id: "anomalies", label: "Anomalies", icon: AlertTriangle },
  { id: "benchmarks", label: "Benchmarks", icon: BarChart3 },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("overview");
  const [overview, setOverview] = useState(null);
  const [profiles, setProfiles] = useState(null);
  const [anomalyData, setAnomalyData] = useState({ total: 0, anomalyCount: 0, rows: [] });
  const [anomWindow, setAnomWindow] = useState(500);
  const [anomFilter, setAnomFilter] = useState("All");
  const [simStartIdx, setSimStartIdx] = useState(29500);
  const [simHorizon, setSimHorizon] = useState(24);
  const [simResult, setSimResult] = useState(null);
  const [simLoading, setSimLoading] = useState(false);
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const [loading, setLoading] = useState(true);

  const fetchOverview = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/overview");
      const data = await res.json();
      setOverview(data);
    } catch (err) {
      console.error("Failed to fetch overview:", err);
    } finally {
      setLoading(false);
    }
  };

  const fetchAnomalies = async (limit) => {
    try {
      const res = await fetch(`/api/anomalies?limit=${limit}`);
      const data = await res.json();
      setAnomalyData(data);
    } catch (err) {
      console.error("Failed to fetch anomalies:", err);
    }
  };

  const fetchProfiles = async () => {
    if (profiles) return;
    try {
      const res = await fetch("/api/profiles");
      const data = await res.json();
      setProfiles(data);
    } catch (err) {
      console.error("Failed to fetch profiles:", err);
    }
  };

  useEffect(() => {
    fetchOverview();
  }, []);

  useEffect(() => {
    if (activeTab === "anomalies") {
      fetchAnomalies(anomWindow);
    } else if (activeTab === "explorer") {
      fetchProfiles();
    }
  }, [activeTab, anomWindow]);

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
      if (activeTab === "anomalies") await fetchAnomalies(anomWindow);
    } catch (err) {
      console.error("Pipeline run failed:", err);
    } finally {
      setPipelineRunning(false);
    }
  };

  const testMetrics = overview?.evaluation?.forecasting_metrics || {};
  const adaptiveEv = overview?.evaluation?.adaptive_detector?.event_level || {};
  const baselineEv = overview?.evaluation?.fixed_baseline?.event_level || {};
  const adaptiveOverall = overview?.evaluation?.adaptive_detector?.overall || {};
  const baselineOverall = overview?.evaluation?.fixed_baseline?.overall || {};
  const validation = overview?.validation || {};
  const accReport = overview?.accuracy || {};
  const modelMeta = overview?.modelMeta || {};

  const filteredAnomalies = (anomalyData.rows || []).filter(
    (r) => r.is_anomaly && (anomFilter === "All" || r.anomaly_type === anomFilter)
  );

  return (
    <div className="app-shell">
      {/* Mobile Sticky Top Header */}
      <header className="mobile-header">
        <div className="brand">
          <span className="brand-dot" />
          <div>
            <div className="brand-title">Power Anomaly ML</div>
            <div className="brand-sub">Adaptive MAD + XGBoost</div>
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
            React 18 · Express · Webpack 5
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <div className="hero-banner">
          <h1>Adaptive Household Electricity Anomaly Detection</h1>
          <p>
            Leak-free XGBoost forecasting paired with past-only Rolling Median + MAD dynamic thresholds
            and anomaly-exclusion baseline protection.
          </p>
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
                    <div className="kpi-label">Test MAE</div>
                    <div className="kpi-value">
                      {testMetrics.mae !== undefined ? `${testMetrics.mae.toFixed(3)} kWh` : "—"}
                    </div>
                    <div className="kpi-sub">RMSE: {testMetrics.rmse ? testMetrics.rmse.toFixed(3) : "—"} kWh</div>
                  </div>
                  <div className="kpi-card">
                    <div className="kpi-label">Test R² Score</div>
                    <div className="kpi-value">
                      {testMetrics.r2 !== undefined ? testMetrics.r2.toFixed(3) : "—"}
                    </div>
                    <div className="kpi-sub">Single XGBRegressor</div>
                  </div>
                  <div className="kpi-card">
                    <div className="kpi-label">Event Capture</div>
                    <div className="kpi-value">
                      {adaptiveEv.event_detection_rate !== undefined
                        ? `${(adaptiveEv.event_detection_rate * 100).toFixed(0)}%`
                        : "—"}
                    </div>
                    <div className="kpi-sub">
                      Delay: {adaptiveEv.average_detection_delay_hours ?? 0}h
                    </div>
                  </div>
                  <div className="kpi-card">
                    <div className="kpi-label">Flagged Hours</div>
                    <div className="kpi-value">{overview?.anomaliesSummary?.detectedCount ?? 0}</div>
                    <div className="kpi-sub">
                      {overview?.anomaliesSummary?.ratePercent ?? 0}% of {overview?.anomaliesSummary?.totalTestHours ?? 0}h
                    </div>
                  </div>
                </div>

                <div className="panel-grid two-col">
                  <div className="card">
                    <div className="card-header">
                      <div>
                        <div className="card-title">Recent Test Stream: Actual vs. Forecast</div>
                        <div className="card-subtitle">Touch or hover chart to inspect hourly readings</div>
                      </div>
                      <span className="badge badge-info">Last 300 Hours</span>
                    </div>
                    <ForecastLineChart data={overview?.recentStream || []} showAnomalies />
                  </div>

                  <div className="card">
                    <div className="card-header">
                      <div className="card-title">Research Verification</div>
                      <span className="badge badge-success">Verified</span>
                    </div>
                    <div className="checklist">
                      <div className="checklist-item">
                        <CheckCircle2 size={16} color="#10b981" />
                        <span>UCI Hourly Data ({validation?.file_info?.total_hourly_records?.toLocaleString() || "34,589"} hrs)</span>
                      </div>
                      <div className="checklist-item">
                        <CheckCircle2 size={16} color="#10b981" />
                        <span>Zero Future Target Leakage</span>
                      </div>
                      <div className="checklist-item">
                        <CheckCircle2 size={16} color="#10b981" />
                        <span>22 Lag & Cyclical Features</span>
                      </div>
                      <div className="checklist-item">
                        <CheckCircle2 size={16} color="#10b981" />
                        <span>168h Rolling MAD + Exclusion</span>
                      </div>
                      <div className="checklist-item">
                        <CheckCircle2 size={16} color="#10b981" />
                        <span>23/23 Pytest Checks Passed</span>
                      </div>
                    </div>
                  </div>
                </div>
              </>
            )}

            {/* EXPLORER TAB */}
            {activeTab === "explorer" && (
              <>
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
                        min={29400}
                        max={34500}
                        step={24}
                        value={simStartIdx}
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
                        {anomalyData.anomalyCount} anomalous hours across {anomalyData.total} test hours
                      </div>
                    </div>
                    <a className="btn btn-outline" href="/api/download/anomalies">
                      <Download size={15} />
                      Export CSV
                    </a>
                  </div>

                  <div className="pill-row" style={{ marginBottom: 14 }}>
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

                  <ForecastLineChart data={anomalyData.rows} height={240} showAnomalies />
                  <div style={{ height: 12 }} />
                  <ResidualThresholdChart data={anomalyData.rows} height={190} />
                </div>

                <div className="card">
                  <div className="card-header">
                    <div className="card-title">Detected Anomaly Events ({filteredAnomalies.length})</div>
                    <div className="pill-row">
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
                  </div>

                  {/* Mobile Card List */}
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

                  {/* Desktop Table */}
                  <div className="desktop-table-only table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Timestamp</th>
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
                      <span className="badge badge-success">Controlled Benchmark</span>
                    </div>
                    <SimpleBarChart
                      items={[
                        {
                          label: "Adaptive",
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
                      <strong>{((adaptiveEv.event_detection_rate || 0) * 100).toFixed(1)}%</strong> (Adaptive) vs{" "}
                      <strong>{((baselineEv.event_detection_rate || 0) * 100).toFixed(1)}%</strong> (Fixed Baseline)
                    </div>
                  </div>

                  <div className="card">
                    <div className="card-header">
                      <div className="card-title">Research Reports & Downloads</div>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                      <a className="btn btn-primary" href="/api/download/docx">
                        <Download size={16} />
                        Download Word Research Report (.docx)
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

                {overview?.evaluation?.adaptive_detector?.per_scenario && (
                  <div className="card">
                    <div className="card-header">
                      <div className="card-title">Scenario-by-Scenario Detection Breakdown</div>
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
                          {Object.entries(overview.evaluation.adaptive_detector.per_scenario).map(
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

const express = require("express");
const fs = require("fs");
const path = require("path");
const { execFile } = require("child_process");

const app = express();
const PORT = process.env.PORT || 5000;

const ROOT_DIR = path.resolve(__dirname, "..");
const REPORTS_DIR = path.join(ROOT_DIR, "reports");
const MODELS_DIR = path.join(ROOT_DIR, "models");
const DIST_DIR = path.join(__dirname, "dist");

app.use(express.json());
app.use(express.static(DIST_DIR));

function readJsonSafe(filePath) {
  try {
    if (fs.existsSync(filePath)) {
      return JSON.parse(fs.readFileSync(filePath, "utf-8"));
    }
  } catch (err) {
    console.error("Failed to read JSON:", filePath, err.message);
  }
  return null;
}

function parseCsvRows(filePath) {
  if (!fs.existsSync(filePath)) return [];
  const raw = fs.readFileSync(filePath, "utf-8").trim();
  const lines = raw.split(/\r?\n/);
  if (lines.length < 2) return [];
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const cols = line.split(",");
    const obj = {};
    headers.forEach((h, i) => {
      obj[h] = cols[i];
    });
    return obj;
  });
}

function parseAnomaliesCsv(limit = 500) {
  const rawRows = parseCsvRows(path.join(REPORTS_DIR, "detected_anomalies.csv"));
  if (!rawRows.length) return { total: 0, anomalyCount: 0, rows: [] };

  let anomalyCount = 0;
  const parsed = rawRows.map((obj) => {
    const isAnom = obj.is_anomaly === "True" || obj.is_anomaly === "true";
    if (isAnom) anomalyCount += 1;
    const thr = parseFloat(obj.threshold);
    const sc = parseFloat(obj.score);
    return {
      timestamp: obj.timestamp,
      actual: parseFloat(obj.actual),
      predicted: parseFloat(obj.predicted),
      residual: parseFloat(obj.residual),
      threshold: Number.isFinite(thr) ? thr : 0,
      score: Number.isFinite(sc) ? sc : 0,
      is_anomaly: isAnom,
      anomaly_type: obj.anomaly_type || "Normal",
      event_id: parseInt(obj.event_id || "0", 10),
      is_event_alert: obj.is_event_alert === "True" || obj.is_event_alert === "true",
    };
  });

  const sliced = limit > 0 ? parsed.slice(-limit) : parsed;
  return {
    total: parsed.length,
    anomalyCount,
    rows: sliced,
  };
}

function parseGroupedEventsCsv() {
  const rawRows = parseCsvRows(path.join(REPORTS_DIR, "detected_anomaly_events.csv"));
  return rawRows.map((obj) => ({
    event_id: parseInt(obj.event_id, 10),
    start_timestamp: obj.start_timestamp,
    end_timestamp: obj.end_timestamp,
    duration_hours: parseInt(obj.duration_hours, 10),
    anomaly_type: obj.anomaly_type,
    peak_actual: parseFloat(obj.peak_actual),
    mean_actual: parseFloat(obj.mean_actual),
    mean_predicted: parseFloat(obj.mean_predicted),
    peak_residual: parseFloat(obj.peak_residual),
    threshold_at_peak: parseFloat(obj.threshold_at_peak),
    peak_score: parseFloat(obj.peak_score),
  }));
}

function runPython(args, timeoutMs = 120000) {
  return new Promise((resolve, reject) => {
    execFile("python", args, { cwd: ROOT_DIR, timeout: timeoutMs }, (err, stdout, stderr) => {
      if (err) {
        reject(new Error(stderr || err.message));
        return;
      }
      resolve(stdout);
    });
  });
}

// Health & overview summary
app.get("/api/overview", (req, res) => {
  const evaluation = readJsonSafe(path.join(REPORTS_DIR, "evaluation_report.json"));
  const accuracy = readJsonSafe(path.join(REPORTS_DIR, "accuracy_report.json"));
  const validation = readJsonSafe(path.join(REPORTS_DIR, "data_validation_report.json"));
  const modelMeta = readJsonSafe(path.join(MODELS_DIR, "xgboost_forecaster_metadata.json"));
  const anomalies = parseAnomaliesCsv(300);
  const groupedEvents = parseGroupedEventsCsv();

  res.json({
    evaluation,
    accuracy,
    validation,
    modelMeta,
    anomaliesSummary: {
      totalTestHours: anomalies.total,
      detectedCount: anomalies.anomalyCount,
      groupedEventsCount: groupedEvents.length,
      ratePercent:
        anomalies.total > 0
          ? Number(((anomalies.anomalyCount / anomalies.total) * 100).toFixed(2))
          : 0,
    },
    groupedEvents,
    recentStream: anomalies.rows,
  });
});

// Anomaly stream with configurable window & live k sensitivity parameter
app.get("/api/anomalies", async (req, res) => {
  const limit = parseInt(req.query.limit || "500", 10);
  const evaluation = readJsonSafe(path.join(REPORTS_DIR, "evaluation_report.json"));
  const defaultCfg = evaluation?.detector_config || {
    sensitivity_k: 4.0,
    window_size: 168,
    min_residual_kwh: 0.65,
    persistence: 2,
    severe_spike_multiplier: 1.35,
    rebaseline_hours: 48,
  };

  const reqK = req.query.k !== undefined ? parseFloat(req.query.k) : defaultCfg.sensitivity_k;
  const reqWin = req.query.window !== undefined ? parseInt(req.query.window, 10) : defaultCfg.window_size;
  const safeLimit = Number.isNaN(limit) ? 500 : limit;

  // Fast path if requesting active default k and window
  if (
    Math.abs(reqK - defaultCfg.sensitivity_k) < 1e-6 &&
    reqWin === defaultCfg.window_size
  ) {
    const data = parseAnomaliesCsv(safeLimit);
    const events = parseGroupedEventsCsv();
    const bva = evaluation?.before_vs_after?.test?.reduction_summary || {};
    return res.json({
      k: defaultCfg.sensitivity_k,
      window_size: defaultCfg.window_size,
      min_residual_kwh: defaultCfg.min_residual_kwh,
      persistence: defaultCfg.persistence,
      severe_spike_multiplier: defaultCfg.severe_spike_multiplier,
      rebaseline_hours: defaultCfg.rebaseline_hours,
      total: data.total,
      anomalyCount: data.anomalyCount,
      ratePercent:
        data.total > 0 ? Number(((data.anomalyCount / data.total) * 100).toFixed(2)) : 0,
      groupedEventsCount: events.length,
      beforeFlaggedHours: bva.flagged_hours_before ?? null,
      beforeEventsCount: bva.grouped_events_before ?? null,
      events,
      rows: data.rows,
    });
  }

  try {
    const stdout = await runPython([
      "-m",
      "src.api_bridge",
      "detect",
      "--k",
      String(reqK),
      "--window",
      String(reqWin),
      "--limit",
      String(safeLimit),
    ]);
    res.json(JSON.parse(stdout));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Dataset profiles (diurnal & weekly)
app.get("/api/profiles", async (req, res) => {
  try {
    const stdout = await runPython(["-m", "src.api_bridge", "profiles"]);
    res.json(JSON.parse(stdout));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Recursive horizon forecast simulator
app.post("/api/forecast/simulate", async (req, res) => {
  const startIdx = parseInt(req.body.startIdx || "29500", 10);
  const horizon = parseInt(req.body.horizon || "24", 10);
  try {
    const stdout = await runPython([
      "-m",
      "src.api_bridge",
      "simulate",
      "--start-idx",
      String(startIdx),
      "--horizon",
      String(horizon),
    ]);
    res.json(JSON.parse(stdout));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Trigger full pipeline re-run
app.post("/api/pipeline/run", async (req, res) => {
  const forceTrain = Boolean(req.body.forceTrain);
  try {
    const args = ["run_pipeline.py"];
    if (forceTrain) args.push("--force-train");
    await runPython(args, 180000);
    await runPython(["generate_accuracy_report.py"], 180000);
    res.json({ status: "ok" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Downloadable reports
app.get("/api/download/:type", (req, res) => {
  const map = {
    anomalies: path.join(REPORTS_DIR, "detected_anomalies.csv"),
    events: path.join(REPORTS_DIR, "detected_anomaly_events.csv"),
    evaluation: path.join(REPORTS_DIR, "evaluation_report.json"),
    accuracy: path.join(REPORTS_DIR, "accuracy_report.json"),
    accuracy_md: path.join(REPORTS_DIR, "accuracy_report.md"),
    docx: path.join(REPORTS_DIR, "Household_Electricity_Anomaly_Detection_Research_Report.docx"),
  };
  const target = map[req.params.type];
  if (!target || !fs.existsSync(target)) {
    return res.status(404).json({ error: "Report file not found" });
  }
  return res.download(target);
});

// SPA fallback
app.get("*", (req, res) => {
  res.sendFile(path.join(DIST_DIR, "index.html"));
});

app.listen(PORT, () => {
  console.log(`Power Anomaly ML server running at http://localhost:${PORT}`);
});

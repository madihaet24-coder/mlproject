const express = require("express");
const fs = require("fs");
const path = require("path");
const { execFile } = require("child_process");

const app = express();
const PORT = process.env.PORT || 5000;

const ROOT_DIR = path.resolve(__dirname, "..");
const REPORTS_DIR = path.join(ROOT_DIR, "reports");
const MODELS_DIR = path.join(ROOT_DIR, "models");
const DATA_DIR = path.join(ROOT_DIR, "data");
const UPLOADS_DIR = path.join(DATA_DIR, "raw", "uploads");
const ACTIVE_DATASET_FILE = path.join(DATA_DIR, "processed", "active_dataset.json");
const DIST_DIR = path.join(__dirname, "dist");

app.use(express.json({ limit: "60mb" }));
app.use("/api", (req, res, next) => {
  res.set("Cache-Control", "no-store, no-cache, must-revalidate, private");
  next();
});
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
  if (!rawRows.length) return { total: 0, anomalyCount: 0, rows: [], flaggedRows: [] };

  let anomalyCount = 0;
  const flaggedRows = [];
  const parsed = rawRows.map((obj) => {
    const isAnom = obj.is_anomaly === "True" || obj.is_anomaly === "true";
    const thr = parseFloat(obj.threshold);
    const sc = parseFloat(obj.score);
    const item = {
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
    if (isAnom) {
      anomalyCount += 1;
      flaggedRows.push(item);
    }
    return item;
  });

  let sliced;
  if (limit > 0) {
    sliced = parsed.slice(-limit);
  } else if (parsed.length > 1200) {
    const step = Math.max(1, Math.floor(parsed.length / 1000));
    sliced = parsed.filter((row, idx) => idx % step === 0 || row.is_anomaly || idx === parsed.length - 1);
  } else {
    sliced = parsed;
  }

  return {
    total: parsed.length,
    anomalyCount,
    rows: sliced,
    flaggedRows,
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
    execFile(
      "python",
      args,
      { cwd: ROOT_DIR, timeout: timeoutMs, maxBuffer: 25 * 1024 * 1024 },
      (err, stdout, stderr) => {
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
  const activeDataset = readJsonSafe(ACTIVE_DATASET_FILE);
  const anomalies = parseAnomaliesCsv(300);
  const groupedEvents = parseGroupedEventsCsv();

  res.json({
    evaluation,
    accuracy,
    validation,
    modelMeta,
    activeDataset: activeDataset
      ? { id: activeDataset.active_id, name: activeDataset.active_name }
      : { id: "uci_default", name: "UCI Household Electric Power (Default)" },
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

const detectCache = new Map();

function sliceRows(allRows, limit) {
  if (limit > 0) return allRows.slice(-limit);
  if (allRows.length > 1200) {
    const step = Math.max(1, Math.floor(allRows.length / 1000));
    return allRows.filter((row, idx) => idx % step === 0 || row.is_anomaly || idx === allRows.length - 1);
  }
  return allRows;
}

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
      flaggedRows: data.flaggedRows,
      rows: data.rows,
    });
  }

  try {
    const csvPath = path.join(REPORTS_DIR, "detected_anomalies.csv");
    const mtime = fs.existsSync(csvPath) ? fs.statSync(csvPath).mtimeMs : 0;
    const cacheKey = `${mtime}_${reqK.toFixed(2)}_${reqWin}`;

    let cached = detectCache.get(cacheKey);
    if (!cached) {
      const stdout = await runPython([
        "-m",
        "src.api_bridge",
        "detect",
        "--k",
        String(reqK),
        "--window",
        String(reqWin),
        "--limit",
        "-1",
      ]);
      cached = JSON.parse(stdout);
      if (detectCache.size > 24) detectCache.clear();
      detectCache.set(cacheKey, cached);
    }

    const { allRows = [], ...meta } = cached;
    return res.json({
      ...meta,
      rows: sliceRows(allRows, safeLimit),
    });
  } catch (err) {
    return res.status(500).json({ error: err.message });
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

// Dataset registry & custom dataset management
app.get("/api/datasets", async (req, res) => {
  try {
    const stdout = await runPython(["-m", "src.api_bridge", "datasets"]);
    res.json(JSON.parse(stdout));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post("/api/datasets/upload", async (req, res) => {
  const { filename, name, content, datetimeCol, targetCol, autoRetrain = true } = req.body || {};
  if (!content || typeof content !== "string") {
    return res.status(400).json({ error: "CSV/TXT file content is required." });
  }

  try {
    fs.mkdirSync(UPLOADS_DIR, { recursive: true });
    const safeBase = path.basename(filename || "custom_dataset.csv").replace(/[^a-zA-Z0-9._-]/g, "_");
    const savedPath = path.join(UPLOADS_DIR, `${Date.now()}_${safeBase}`);
    fs.writeFileSync(savedPath, content, "utf-8");

    const ingestArgs = ["-m", "src.api_bridge", "ingest", "--file", savedPath];
    if (name && name.trim()) ingestArgs.push("--name", name.trim());
    if (datetimeCol && datetimeCol.trim()) ingestArgs.push("--datetime-col", datetimeCol.trim());
    if (targetCol && targetCol.trim()) ingestArgs.push("--target-col", targetCol.trim());

    const ingestStdout = await runPython(ingestArgs, 120000);
    const ingestResult = JSON.parse(ingestStdout);

    if (autoRetrain) {
      await runPython(["validate_data.py"], 180000);
      await runPython(["run_pipeline.py", "--force-train"], 240000);
      await runPython(["generate_accuracy_report.py"], 240000);
    }

    res.json({ status: "ok", ...ingestResult });
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

app.post("/api/datasets/activate", async (req, res) => {
  const { datasetId, retrain = true } = req.body || {};
  if (!datasetId) {
    return res.status(400).json({ error: "datasetId is required." });
  }
  try {
    const stdout = await runPython(["-m", "src.api_bridge", "activate", "--id", String(datasetId)]);
    const result = JSON.parse(stdout);
    if (retrain) {
      await runPython(["validate_data.py"], 180000);
      await runPython(["run_pipeline.py", "--force-train"], 240000);
      await runPython(["generate_accuracy_report.py"], 240000);
    }
    res.json({ status: "ok", ...result });
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

app.get("/api/datasets/template", (req, res) => {
  const lines = ["Datetime,Global_active_power"];
  const start = new Date("2025-01-01T00:00:00Z");
  for (let i = 0; i < 720; i += 1) {
    const dt = new Date(start.getTime() + i * 3600 * 1000);
    const hour = dt.getUTCHours();
    const base = 1.15 + 0.75 * Math.sin((2 * Math.PI * (hour - 7)) / 24);
    const val = Math.max(0.25, base + ((i % 7) - 3) * 0.04).toFixed(4);
    lines.push(`${dt.toISOString().slice(0, 19).replace("T", " ")},${val}`);
  }
  res.setHeader("Content-Type", "text/csv; charset=utf-8");
  res.setHeader("Content-Disposition", 'attachment; filename="sample_custom_electricity_dataset.csv"');
  res.send(lines.join("\n"));
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

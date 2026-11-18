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

function parseAnomaliesCsv(limit = 500) {
  const csvPath = path.join(REPORTS_DIR, "detected_anomalies.csv");
  if (!fs.existsSync(csvPath)) return { total: 0, anomalyCount: 0, rows: [] };

  const raw = fs.readFileSync(csvPath, "utf-8").trim();
  const lines = raw.split(/\r?\n/);
  if (lines.length < 2) return { total: 0, anomalyCount: 0, rows: [] };

  const headers = lines[0].split(",");
  const dataLines = lines.slice(1);
  let anomalyCount = 0;

  const parsed = dataLines.map((line) => {
    const cols = line.split(",");
    const obj = {};
    headers.forEach((h, i) => {
      obj[h] = cols[i];
    });
    const isAnom = obj.is_anomaly === "True" || obj.is_anomaly === "true";
    if (isAnom) anomalyCount += 1;
    return {
      timestamp: obj.timestamp,
      actual: parseFloat(obj.actual),
      predicted: parseFloat(obj.predicted),
      residual: parseFloat(obj.residual),
      threshold: parseFloat(obj.threshold),
      score: parseFloat(obj.score),
      is_anomaly: isAnom,
      anomaly_type: obj.anomaly_type || "Normal",
    };
  });

  const sliced = limit > 0 ? parsed.slice(-limit) : parsed;
  return {
    total: parsed.length,
    anomalyCount,
    rows: sliced,
  };
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

  res.json({
    evaluation,
    accuracy,
    validation,
    modelMeta,
    anomaliesSummary: {
      totalTestHours: anomalies.total,
      detectedCount: anomalies.anomalyCount,
      ratePercent: anomalies.total > 0 ? Number(((anomalies.anomalyCount / anomalies.total) * 100).toFixed(2)) : 0,
    },
    recentStream: anomalies.rows,
  });
});

// Anomaly stream with configurable window
app.get("/api/anomalies", (req, res) => {
  const limit = parseInt(req.query.limit || "500", 10);
  const data = parseAnomaliesCsv(Number.isNaN(limit) ? 500 : limit);
  res.json(data);
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

import React, { useState } from "react";

function buildPath(points, xScale, yScale) {
  if (!points.length) return "";
  return points
    .map((pt, idx) => `${idx === 0 ? "M" : "L"}${xScale(idx).toFixed(1)},${yScale(pt).toFixed(1)}`)
    .join(" ");
}

export function ForecastLineChart({ data, height = 270, showAnomalies = false }) {
  const [hoverIdx, setHoverIdx] = useState(null);

  if (!data || data.length === 0) {
    return <div className="card-subtitle">No time-series data available.</div>;
  }

  const width = 760;
  const pad = { top: 18, right: 16, bottom: 28, left: 44 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;

  const actualVals = data.map((d) => Number(d.actual ?? d.actual_kwh ?? 0));
  const predVals = data.map((d) => Number(d.predicted ?? d.forecast_kwh ?? 0));
  const maxY = Math.max(0.5, ...actualVals, ...predVals) * 1.1;

  const xScale = (i) => pad.left + (i / Math.max(1, data.length - 1)) * innerW;
  const yScale = (v) => pad.top + innerH - (v / maxY) * innerH;

  const actualPath = buildPath(actualVals, xScale, yScale);
  const predPath = buildPath(predVals, xScale, yScale);

  const handlePointer = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const relX = ((e.clientX - rect.left) / rect.width) * width - pad.left;
    const idx = Math.round((relX / innerW) * (data.length - 1));
    if (idx >= 0 && idx < data.length) setHoverIdx(idx);
  };

  const active = hoverIdx !== null ? data[hoverIdx] : null;

  return (
    <div className="chart-container">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        style={{ width: "100%", height: "auto", display: "block", touchAction: "pan-y" }}
        onMouseMove={handlePointer}
        onMouseLeave={() => setHoverIdx(null)}
        onTouchMove={(e) => e.touches[0] && handlePointer(e.touches[0])}
      >
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const yVal = maxY * t;
          const y = yScale(yVal);
          return (
            <g key={t}>
              <line x1={pad.left} x2={width - pad.right} y1={y} y2={y} stroke="#e2e8f0" strokeWidth="1" />
              <text x={pad.left - 8} y={y + 4} fontSize="10" fill="#64748b" textAnchor="end">
                {yVal.toFixed(1)}
              </text>
            </g>
          );
        })}

        <path d={actualPath} fill="none" stroke="#0f3e72" strokeWidth="2" />
        <path d={predPath} fill="none" stroke="#0ea5e9" strokeWidth="1.8" strokeDasharray="5 3" />

        {showAnomalies &&
          data.map((d, idx) =>
            d.is_anomaly ? (
              <circle
                key={idx}
                cx={xScale(idx)}
                cy={yScale(actualVals[idx])}
                r="4.5"
                fill="#ef4444"
                stroke="#ffffff"
                strokeWidth="1.5"
              />
            ) : null
          )}

        {hoverIdx !== null && (
          <line
            x1={xScale(hoverIdx)}
            x2={xScale(hoverIdx)}
            y1={pad.top}
            y2={height - pad.bottom}
            stroke="#94a3b8"
            strokeDasharray="3 3"
          />
        )}

        <text x={pad.left} y={height - 6} fontSize="10" fill="#64748b">
          {String(data[0].timestamp || "").slice(0, 16)}
        </text>
        <text x={width - pad.right} y={height - 6} fontSize="10" fill="#64748b" textAnchor="end">
          {String(data[data.length - 1].timestamp || "").slice(0, 16)}
        </text>
      </svg>

      {active && (
        <div className="chart-tooltip" style={{ top: 10, right: 12 }}>
          <div><strong>{String(active.timestamp).slice(0, 16)}</strong></div>
          <div>Actual: {Number(active.actual ?? active.actual_kwh).toFixed(3)} kWh</div>
          <div>Forecast: {Number(active.predicted ?? active.forecast_kwh).toFixed(3)} kWh</div>
          {active.is_anomaly && <div style={{ color: "#fca5a5" }}>Anomaly: {active.anomaly_type}</div>}
        </div>
      )}
    </div>
  );
}

export function ResidualThresholdChart({ data, height = 210 }) {
  const [hoverIdx, setHoverIdx] = useState(null);
  if (!data || data.length === 0) return null;

  const width = 760;
  const pad = { top: 14, right: 16, bottom: 26, left: 44 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;

  const resVals = data.map((d) => Number(d.residual || 0));
  const thrVals = data.map((d) => Number(d.threshold || 0));
  const maxY = Math.max(0.5, ...resVals, ...thrVals) * 1.1;

  const xScale = (i) => pad.left + (i / Math.max(1, data.length - 1)) * innerW;
  const yScale = (v) => pad.top + innerH - (v / maxY) * innerH;

  const handlePointer = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const relX = ((e.clientX - rect.left) / rect.width) * width - pad.left;
    const idx = Math.round((relX / innerW) * (data.length - 1));
    if (idx >= 0 && idx < data.length) setHoverIdx(idx);
  };

  const active = hoverIdx !== null ? data[hoverIdx] : null;

  return (
    <div className="chart-container">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        style={{ width: "100%", height: "auto", display: "block", touchAction: "pan-y" }}
        onMouseMove={handlePointer}
        onMouseLeave={() => setHoverIdx(null)}
        onTouchMove={(e) => e.touches[0] && handlePointer(e.touches[0])}
      >
        {[0, 0.5, 1].map((t) => {
          const yVal = maxY * t;
          const y = yScale(yVal);
          return (
            <g key={t}>
              <line x1={pad.left} x2={width - pad.right} y1={y} y2={y} stroke="#e2e8f0" strokeWidth="1" />
              <text x={pad.left - 8} y={y + 4} fontSize="10" fill="#64748b" textAnchor="end">
                {yVal.toFixed(1)}
              </text>
            </g>
          );
        })}

        <path d={buildPath(resVals, xScale, yScale)} fill="none" stroke="#64748b" strokeWidth="1.5" />
        <path d={buildPath(thrVals, xScale, yScale)} fill="none" stroke="#f59e0b" strokeWidth="2.2" />

        {hoverIdx !== null && (
          <line
            x1={xScale(hoverIdx)}
            x2={xScale(hoverIdx)}
            y1={pad.top}
            y2={height - pad.bottom}
            stroke="#94a3b8"
            strokeDasharray="3 3"
          />
        )}
      </svg>
      {active && (
        <div className="chart-tooltip" style={{ top: 8, right: 12 }}>
          <div><strong>{String(active.timestamp).slice(0, 16)}</strong></div>
          <div>Residual: {Number(active.residual).toFixed(3)} kWh</div>
          <div>MAD Threshold: {Number(active.threshold).toFixed(3)} kWh</div>
        </div>
      )}
    </div>
  );
}

export function SimpleBarChart({ items, labelKey, valueKey, color = "#0284c7", suffix = "" }) {
  if (!items || items.length === 0) return null;
  const maxVal = Math.max(0.01, ...items.map((d) => Number(d[valueKey] || 0)));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {items.map((item, i) => {
        const val = Number(item[valueKey] || 0);
        const pct = Math.min(100, (val / maxVal) * 100);
        return (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ width: 64, fontSize: 12, fontWeight: 600, color: "#475569" }}>
              {item[labelKey]}
            </div>
            <div style={{ flex: 1, background: "#f1f5f9", borderRadius: 999, height: 12, overflow: "hidden" }}>
              <div
                style={{
                  width: `${pct}%`,
                  height: "100%",
                  background: item.color || color,
                  borderRadius: 999,
                  transition: "width 0.3s ease",
                }}
              />
            </div>
            <div style={{ width: 68, textAlign: "right", fontSize: 12, fontWeight: 700, fontFamily: "JetBrains Mono, monospace" }}>
              {val.toFixed(2)}{suffix}
            </div>
          </div>
        );
      })}
    </div>
  );
}

/**
 * static/js/visualizer.js
 * Obsidian Quant & Emerald-Gold Visualization Engine for CreditRisk-Neural-Bank.
 */
window.Visualizer = (function () {
  const charts = {
    ks: null,
    decile: null,
    roc: null,
    weights: null,
    loss: null
  };

  let lastGrid = null;
  let lastClassNames = null;
  let activeBorrowerPoint = null;

  const darkChartDefaults = {
    color: "#94a3b8",
    gridColor: "rgba(148, 163, 184, 0.10)",
    tickColor: "#94a3b8"
  };

  function renderBoundaryCanvas(grid, classNames, borrowerPoint = null) {
    const canvas = document.getElementById("boundaryCanvas");
    if (!canvas || !grid) return;
    lastGrid = grid;
    lastClassNames = classNames;
    if (borrowerPoint) activeBorrowerPoint = borrowerPoint;

    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Deep obsidian base
    ctx.fillStyle = "#050811";
    ctx.fillRect(0, 0, W, H);

    const res = grid.resolution;
    const cellW = W / res;
    const cellH = H / res;
    const cutoff = grid.threshold || 0.35;

    for (let r = 0; r < res; r++) {
      const rowIdx = res - 1 - r;
      for (let c = 0; c < res; c++) {
        const pd = grid.probs[rowIdx][c];
        if (pd < cutoff * 0.65) {
          // Stage 1 Prime Performing Zone (Emerald / Cyan-Teal)
          const intensity = Math.min(1, (cutoff * 0.65 - pd) / Math.max(0.05, cutoff * 0.65));
          ctx.fillStyle = `rgba(16, 185, 129, ${0.18 + intensity * 0.44})`;
        } else if (pd < cutoff) {
          // Stage 2 SICR Watch Zone (Champagne Gold / Amber)
          const intensity = (pd - cutoff * 0.65) / Math.max(0.05, cutoff * 0.35);
          ctx.fillStyle = `rgba(245, 158, 11, ${0.20 + intensity * 0.38})`;
        } else {
          // Stage 3 / Declined High Default Risk Zone (Crimson / Magenta-Rose)
          const intensity = Math.min(1, (pd - cutoff) / Math.max(0.05, 1.0 - cutoff));
          ctx.fillStyle = `rgba(244, 63, 94, ${0.24 + intensity * 0.54})`;
        }
        ctx.fillRect(c * cellW, r * cellH, cellW + 0.8, cellH + 0.8);

        // Underwriting PD Cutoff Contour (Luminous Gold)
        if (Math.abs(pd - cutoff) < 0.022) {
          ctx.fillStyle = "rgba(251, 191, 36, 0.95)";
          ctx.fillRect(c * cellW, r * cellH, cellW + 1.2, cellH + 1.2);
        }
      }
    }

    // Terminal radar grid lines
    ctx.strokeStyle = "rgba(16, 185, 129, 0.14)";
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
      ctx.beginPath();
      ctx.moveTo((W / 4) * i, 0);
      ctx.lineTo((W / 4) * i, H);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(0, (H / 4) * i);
      ctx.lineTo(W, (H / 4) * i);
      ctx.stroke();
    }

    const spanX = (grid.x_max - grid.x_min) || 1;
    const spanY = (grid.y_max - grid.y_min) || 1;

    // Plot loan portfolio facilities
    grid.points.forEach((pt) => {
      const cx = ((pt.x - grid.x_min) / spanX) * W;
      const cy = H - ((pt.y - grid.y_min) / spanY) * H;
      ctx.beginPath();
      ctx.arc(cx, cy, 4.0, 0, 2 * Math.PI);
      ctx.fillStyle = pt.label === 1 ? "#fb7185" : "#34d399";
      ctx.strokeStyle = "#020617";
      ctx.lineWidth = 1.3;
      ctx.fill();
      ctx.stroke();
    });

    // Draw Live Single-Borrower Underwriting Crosshair Marker
    if (activeBorrowerPoint && activeBorrowerPoint.x !== undefined && activeBorrowerPoint.y !== undefined) {
      const bx = Math.max(8, Math.min(W - 8, ((activeBorrowerPoint.x - grid.x_min) / spanX) * W));
      const by = Math.max(8, Math.min(H - 8, H - ((activeBorrowerPoint.y - grid.y_min) / spanY) * H));

      ctx.strokeStyle = "rgba(251, 191, 36, 0.80)";
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(bx, 0);
      ctx.lineTo(bx, H);
      ctx.moveTo(0, by);
      ctx.lineTo(W, by);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.beginPath();
      ctx.arc(bx, by, 8.0, 0, 2 * Math.PI);
      ctx.fillStyle = "#fbbf24";
      ctx.strokeStyle = "#020617";
      ctx.lineWidth = 2.4;
      ctx.fill();
      ctx.stroke();
    }

    if (classNames && classNames.length >= 2) {
      const l0 = document.getElementById("legendClass0");
      const l1 = document.getElementById("legendClass1");
      if (l0) l0.textContent = classNames[0];
      if (l1) l1.textContent = classNames[1];
    }
  }

  function updateBorrowerMarker(xVal, yVal) {
    activeBorrowerPoint = { x: xVal, y: yVal };
    if (lastGrid) {
      renderBoundaryCanvas(lastGrid, lastClassNames, activeBorrowerPoint);
    }
  }

  function renderKSChart(ksCurve, ksStat, ksOptTh) {
    const ctx = document.getElementById("canvasKSChart");
    if (!ctx || typeof Chart === "undefined") return;
    if (charts.ks) charts.ks.destroy();

    charts.ks = new Chart(ctx, {
      type: "line",
      data: {
        labels: ksCurve.map((pt) => `${Math.round(pt.threshold * 100)}%`),
        datasets: [
          {
            label: "Cumulative Defaults Caught (Bad %)",
            data: ksCurve.map((pt) => pt.cum_bad),
            borderColor: "#fb7185",
            backgroundColor: "rgba(251, 113, 133, 0.12)",
            fill: false,
            tension: 0.25,
            borderWidth: 2.3,
            pointRadius: 1.5
          },
          {
            label: "Cumulative Performing Flagged (Good %)",
            data: ksCurve.map((pt) => pt.cum_good),
            borderColor: "#34d399",
            backgroundColor: "rgba(52, 211, 153, 0.10)",
            fill: false,
            tension: 0.25,
            borderWidth: 2.3,
            pointRadius: 1.5
          },
          {
            label: `KS Separation Gap (Max KS = ${(ksStat * 100).toFixed(1)}% @ ${Math.round(ksOptTh * 100)}%)`,
            data: ksCurve.map((pt) => pt.ks_gap),
            borderColor: "#fbbf24",
            backgroundColor: "rgba(251, 191, 36, 0.16)",
            borderDash: [4, 3],
            fill: true,
            tension: 0.25,
            borderWidth: 2,
            pointRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: darkChartDefaults.color } } },
        scales: {
          x: {
            title: { display: true, text: "PD Cutoff Threshold (%)", color: darkChartDefaults.color },
            ticks: { color: darkChartDefaults.tickColor, maxTicksLimit: 10 },
            grid: { color: darkChartDefaults.gridColor }
          },
          y: {
            min: 0,
            max: 100,
            title: { display: true, text: "Cumulative Capture / Separation (%)", color: darkChartDefaults.color },
            ticks: { color: darkChartDefaults.tickColor },
            grid: { color: darkChartDefaults.gridColor }
          }
        }
      }
    });
  }

  function renderDecileChart(decileTable) {
    const ctx = document.getElementById("canvasDecileChart");
    if (!ctx || typeof Chart === "undefined") return;
    if (charts.decile) charts.decile.destroy();

    charts.decile = new Chart(ctx, {
      type: "bar",
      data: {
        labels: decileTable.map((d) => d.decile),
        datasets: [
          {
            type: "bar",
            label: "Observed Default Rate (%)",
            data: decileTable.map((d) => d.observed_bad_pct),
            backgroundColor: "rgba(244, 63, 94, 0.75)",
            borderRadius: 4,
            yAxisID: "y"
          },
          {
            type: "line",
            label: "Model Predicted PD (%)",
            data: decileTable.map((d) => d.predicted_pd_pct),
            borderColor: "#22d3ee",
            backgroundColor: "#22d3ee",
            borderWidth: 2.5,
            pointRadius: 3,
            tension: 0.25,
            yAxisID: "y"
          },
          {
            type: "line",
            label: "Expected Credit Loss ($k)",
            data: decileTable.map((d) => d.ecl_k),
            borderColor: "#fbbf24",
            borderDash: [4, 3],
            borderWidth: 2,
            pointRadius: 2.5,
            tension: 0.25,
            yAxisID: "y1"
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: darkChartDefaults.color } } },
        scales: {
          x: {
            ticks: { color: darkChartDefaults.tickColor },
            grid: { color: darkChartDefaults.gridColor }
          },
          y: {
            type: "linear",
            position: "left",
            title: { display: true, text: "Default Rate / PD (%)", color: darkChartDefaults.color },
            ticks: { color: darkChartDefaults.tickColor },
            grid: { color: darkChartDefaults.gridColor }
          },
          y1: {
            type: "linear",
            position: "right",
            grid: { drawOnChartArea: false },
            title: { display: true, text: "Expected Credit Loss ($k)", color: darkChartDefaults.color },
            ticks: { color: darkChartDefaults.tickColor }
          }
        }
      }
    });
  }

  function renderROCChart(rocCurve, aucVal, giniVal) {
    const ctx = document.getElementById("canvasROCChart");
    if (!ctx || typeof Chart === "undefined") return;
    if (charts.roc) charts.roc.destroy();

    charts.roc = new Chart(ctx, {
      type: "line",
      data: {
        datasets: [
          {
            label: `Credit Model ROC (AUC = ${aucVal.toFixed(3)} | Gini = ${giniVal.toFixed(3)})`,
            data: rocCurve,
            borderColor: "#10b981",
            backgroundColor: "rgba(16, 185, 129, 0.16)",
            fill: true,
            pointRadius: 1.5,
            borderWidth: 2.4
          },
          {
            label: "Random Uninformed Baseline (Gini = 0.000)",
            data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            borderColor: "#64748b",
            borderDash: [5, 5],
            pointRadius: 0,
            borderWidth: 1.5
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: darkChartDefaults.color } } },
        scales: {
          x: {
            type: "linear",
            min: 0,
            max: 1,
            title: { display: true, text: "False Alarm Rate (Rejected Good Borrowers)", color: darkChartDefaults.color },
            ticks: { color: darkChartDefaults.tickColor },
            grid: { color: darkChartDefaults.gridColor }
          },
          y: {
            type: "linear",
            min: 0,
            max: 1,
            title: { display: true, text: "Default Capture Rate (True Positive Rate)", color: darkChartDefaults.color },
            ticks: { color: darkChartDefaults.tickColor },
            grid: { color: darkChartDefaults.gridColor }
          }
        }
      }
    });
  }

  function renderWeightsChart(featureNames, weights, ivScores) {
    const ctx = document.getElementById("canvasWeightsChart");
    if (!ctx || typeof Chart === "undefined") return;
    if (charts.weights) charts.weights.destroy();

    charts.weights = new Chart(ctx, {
      type: "bar",
      data: {
        labels: featureNames,
        datasets: [
          {
            label: "Signed Risk Driver Weight (+ Default Risk / - Credit Buffer)",
            data: weights,
            backgroundColor: weights.map((w) => w >= 0 ? "rgba(244, 63, 94, 0.82)" : "rgba(16, 185, 129, 0.82)"),
            borderRadius: 4
          },
          {
            label: "Information Value (IV Predictive Power)",
            data: ivScores || weights.map((w) => Math.abs(w)),
            backgroundColor: "rgba(245, 158, 11, 0.78)",
            borderRadius: 4
          }
        ]
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: darkChartDefaults.color } } },
        scales: {
          x: {
            ticks: { color: darkChartDefaults.tickColor },
            grid: { color: darkChartDefaults.gridColor }
          },
          y: {
            ticks: { color: darkChartDefaults.tickColor },
            grid: { color: darkChartDefaults.gridColor }
          }
        }
      }
    });
  }

  function renderLossChart(lossHistory) {
    const ctx = document.getElementById("canvasLossChart");
    if (!ctx || typeof Chart === "undefined") return;
    if (charts.loss) charts.loss.destroy();

    charts.loss = new Chart(ctx, {
      type: "line",
      data: {
        labels: lossHistory.map((item) => `Ep ${item.epoch}`),
        datasets: [{
          label: "Cost-Sensitive Credit Cross-Entropy Loss",
          data: lossHistory.map((item) => item.loss),
          borderColor: "#22d3ee",
          backgroundColor: "rgba(34, 211, 238, 0.14)",
          fill: true,
          tension: 0.25,
          pointRadius: lossHistory.length > 45 ? 0 : 2,
          borderWidth: 2.2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: darkChartDefaults.color } } },
        scales: {
          x: {
            ticks: { color: darkChartDefaults.tickColor, maxTicksLimit: 8 },
            grid: { color: darkChartDefaults.gridColor }
          },
          y: {
            ticks: { color: darkChartDefaults.tickColor },
            grid: { color: darkChartDefaults.gridColor }
          }
        }
      }
    });
  }

  return {
    renderBoundaryCanvas,
    updateBorrowerMarker,
    renderKSChart,
    renderDecileChart,
    renderROCChart,
    renderWeightsChart,
    renderLossChart
  };
})();

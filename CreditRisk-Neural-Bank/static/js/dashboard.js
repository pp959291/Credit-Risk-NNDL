/**
 * static/js/dashboard.js
 * Main UI Controller for CreditRisk-Neural-Bank (Obsidian Emerald-Gold Terminal Edition).
 */
(function () {
  const state = {
    dataset: null,
    rawCSVText: null,
    activePredictor: null,
    lastData: null,
    axisX: 0,
    axisY: 1,
    backendAvailable: false
  };

  function setStatus(text, isBusy = false) {
    const dot = document.getElementById("statusDot");
    const lbl = document.getElementById("statusText");
    if (lbl) lbl.textContent = text;
    if (dot) {
      dot.className = isBusy
        ? "w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse shadow-[0_0_8px_#fbbf24]"
        : "w-2.5 h-2.5 rounded-full bg-emerald-400 shadow-[0_0_8px_#34d399]";
    }
  }

  function getConfigFromUI() {
    return {
      modelType: document.getElementById("selectModelType").value,
      lr: parseFloat(document.getElementById("inputLR").value),
      epochs: parseInt(document.getElementById("inputEpochs").value, 10),
      hiddenDim: parseInt(document.getElementById("inputHidden").value, 10),
      treeDepth: parseInt(document.getElementById("inputDepth").value, 10),
      threshold: parseFloat(document.getElementById("inputThreshold").value) / 100.0,
      lgdPct: parseFloat(document.getElementById("inputLGD").value),
      eadK: parseFloat(document.getElementById("inputEAD").value),
      standardize: document.getElementById("chkStandardize").checked,
      axisX: state.axisX,
      axisY: state.axisY
    };
  }

  async function postBackendAPI(endpoint, payload) {
    const apiUrl = `${window.location.origin}${endpoint}`;
    const res = await fetch(apiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error("Backend request failed");
    return res.json();
  }

  async function runTraining() {
    if (!state.dataset) return;
    setStatus("Calibrating Credit Risk Engine...", true);

    const cfg = getConfigFromUI();
    const modelNames = {
      woe_scorecard: "ENGINE: WOE SCORECARD (PDO 300-850)",
      credit_neural_mlp: "ENGINE: CREDIT NEURAL MLP (2-LAYER)",
      credit_interaction_net: "ENGINE: LEVERAGE INTERACTION NET",
      credit_gbdt: "ENGINE: GRADIENT BOOSTED CREDIT TREES"
    };
    const badgeEl = document.getElementById("activeModelBadge");
    if (badgeEl) badgeEl.textContent = modelNames[cfg.modelType] || cfg.modelType;

    if (state.backendAvailable) {
      try {
        const json = await postBackendAPI("/api/train", {
          model_type: cfg.modelType,
          hyperparams: {
            learning_rate: cfg.lr,
            epochs: cfg.epochs,
            hidden_dim: cfg.hiddenDim,
            max_depth: cfg.treeDepth,
            threshold: cfg.threshold,
            lgd: cfg.lgdPct / 100.0,
            ead_k: cfg.eadK
          },
          standardize: cfg.standardize,
          axis_x: cfg.axisX,
          axis_y: cfg.axisY
        });
        if (json.status === "ok") {
          state.lastData = json.data;
          updateDashboardUI(json.data);
          setStatus(`ONLINE · GINI ${json.data.metrics.gini_coefficient.toFixed(3)} · KS ${(json.data.metrics.ks_statistic * 100).toFixed(1)}%`);
          return;
        }
      } catch (_) {
        state.backendAvailable = false;
      }
    }

    const result = window.ModelTrainer.trainPipeline(state.dataset, cfg);
    state.activePredictor = result.predictSingle;
    state.lastData = result.responsePayload;
    updateDashboardUI(result.responsePayload);
    setStatus(`ONLINE · GINI ${result.responsePayload.metrics.gini_coefficient.toFixed(3)} · KS ${(result.responsePayload.metrics.ks_statistic * 100).toFixed(1)}%`);
  }

  async function generateNewCreditPortfolio() {
    const preset = document.getElementById("selectPreset").value;
    const samples = parseInt(document.getElementById("inputSamples").value, 10);
    const stress = parseFloat(document.getElementById("inputNoise").value);

    state.dataset = window.DataGenerator.generateSynthetic(preset, samples, stress);
    state.axisX = 0;
    state.axisY = Math.min(1, state.dataset.featureNames.length - 1);

    if (state.dataset.meta) {
      const lgdInput = document.getElementById("inputLGD");
      const eadInput = document.getElementById("inputEAD");
      if (lgdInput) {
        lgdInput.value = state.dataset.meta.defaultLGD;
        document.getElementById("valLGD").textContent = `${state.dataset.meta.defaultLGD}%`;
      }
      if (eadInput) {
        eadInput.value = state.dataset.meta.avgEAD;
        document.getElementById("valEAD").textContent = `$${state.dataset.meta.avgEAD}k`;
      }
    }

    if (state.backendAvailable) {
      try {
        const cfg = getConfigFromUI();
        const json = await postBackendAPI("/api/generate", {
          preset,
          n_samples: samples,
          noise: stress,
          model_type: cfg.modelType,
          hyperparams: {
            learning_rate: cfg.lr,
            epochs: cfg.epochs,
            hidden_dim: cfg.hiddenDim,
            max_depth: cfg.treeDepth,
            threshold: cfg.threshold,
            lgd: cfg.lgdPct / 100.0,
            ead_k: cfg.eadK
          },
          standardize: cfg.standardize
        });
        if (json.status === "ok") {
          state.lastData = json.data;
          updateDashboardUI(json.data);
          setStatus(`ONLINE · GINI ${json.data.metrics.gini_coefficient.toFixed(3)} · KS ${(json.data.metrics.ks_statistic * 100).toFixed(1)}%`);
          return;
        }
      } catch (_) {
        state.backendAvailable = false;
      }
    }

    await runTraining();
  }

  function updateDashboardUI(data) {
    const ds = data.dataset_summary;
    const m = data.metrics;
    const b = data.basel_summary;

    const badRatePct = ((ds.class_counts.class_1 / Math.max(1, ds.total_rows)) * 100).toFixed(1);
    document.getElementById("summaryRows").textContent = ds.total_rows;
    document.getElementById("summaryFeatures").textContent = ds.feature_names.length;
    document.getElementById("summaryClasses").textContent = `Perf: ${ds.class_counts.class_0} | Def: ${ds.class_counts.class_1} (${badRatePct}%)`;

    // Top Basel III Executive Telemetry Ribbon
    document.getElementById("kpiPortfolioECL").textContent = `$${b.ecl.toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 })}k`;
    document.getElementById("kpiPortfolioEAD").textContent = `Total EAD: $${b.ead.toLocaleString(undefined, { maximumFractionDigits: 0 })}k · Avg PD: ${(b.pd * 100).toFixed(1)}%`;
    document.getElementById("kpiBaselRWA").textContent = `$${b.rwa.toLocaleString(undefined, { maximumFractionDigits: 0 })}k`;
    document.getElementById("kpiRegCapital").textContent = `8% Tier-1 Cap: $${b.regulatory_capital_8pct.toFixed(1)}k (RW: ${b.risk_weight_pct}%)`;
    document.getElementById("kpiGiniKS").textContent = `${m.gini_coefficient.toFixed(3)} / ${(m.ks_statistic * 100).toFixed(1)}%`;
    document.getElementById("kpiKSOpt").textContent = `Optimal KS Cutoff: ${Math.round(m.ks_optimal_threshold * 100)}% PD`;
    const pnlSign = b.net_portfolio_pnl_k >= 0 ? "+" : "";
    document.getElementById("kpiNetPnL").textContent = `${pnlSign}$${b.net_portfolio_pnl_k.toFixed(1)}k`;
    document.getElementById("kpiLossSaved").textContent = `Default Loss Avoided: $${b.loss_prevented_k.toFixed(1)}k`;

    // Model Validation Telemetry Strip
    document.getElementById("metricAUC").textContent = m.roc_auc.toFixed(3);
    document.getElementById("metricBrier").textContent = `Brier Score: ${m.brier_score.toFixed(4)}`;
    document.getElementById("metricGini").textContent = m.gini_coefficient.toFixed(3);
    document.getElementById("metricAcc").textContent = `Test Acc: ${(m.test_accuracy * 100).toFixed(1)}% (Train: ${(m.train_accuracy * 100).toFixed(1)}%)`;
    document.getElementById("metricKS").textContent = `${(m.ks_statistic * 100).toFixed(1)}%`;
    document.getElementById("metricKSTh").textContent = `Max Sep @ PD = ${Math.round(m.ks_optimal_threshold * 100)}%`;
    document.getElementById("metricRecall").textContent = `${(m.recall * 100).toFixed(1)}%`;
    document.getElementById("metricF1Prec").textContent = `F1: ${m.f1_score.toFixed(3)} · Prec: ${m.precision.toFixed(3)}`;

    // Underwriting Financial Matrix & IFRS 9 Stages
    document.getElementById("cmTN").textContent = m.confusion_matrix.tn;
    document.getElementById("cmFP").textContent = m.confusion_matrix.fp;
    document.getElementById("cmFN").textContent = m.confusion_matrix.fn;
    document.getElementById("cmTP").textContent = m.confusion_matrix.tp;

    document.getElementById("cmInterestEarned").textContent = `+$${b.net_interest_earned_k.toFixed(1)}k Interest`;
    document.getElementById("cmOppCost").textContent = `-$${b.opp_cost_rejected_k.toFixed(1)}k Foregone`;
    document.getElementById("cmLossIncurred").textContent = `-$${b.credit_loss_incurred_k.toFixed(1)}k LGD Loss`;
    document.getElementById("cmLossPrevented").textContent = `+$${b.loss_prevented_k.toFixed(1)}k Saved`;

    document.getElementById("stage1Count").textContent = b.stage_1_count;
    document.getElementById("stage2Count").textContent = b.stage_2_count;
    document.getElementById("stage3Count").textContent = b.stage_3_count;

    populateAxes(ds.feature_names, data.boundary_grid.axis_x, data.boundary_grid.axis_y);
    window.Visualizer.renderBoundaryCanvas(data.boundary_grid, ds.class_names);
    window.Visualizer.renderKSChart(data.ks_curve, m.ks_statistic, m.ks_optimal_threshold);
    window.Visualizer.renderDecileChart(data.decile_table);
    window.Visualizer.renderROCChart(data.roc_curve, m.roc_auc, m.gini_coefficient);
    window.Visualizer.renderWeightsChart(ds.feature_names, data.weights, data.iv_scores);
    window.Visualizer.renderLossChart(data.loss_history);

    renderPreviewTable(ds.feature_names, ds.target_name, ds.preview_rows);
    renderInferenceSliders(data.feature_ranges);
  }

  function populateAxes(featureNames, axX, axY) {
    const selX = document.getElementById("selectAxisX");
    const selY = document.getElementById("selectAxisY");
    selX.innerHTML = "";
    selY.innerHTML = "";

    featureNames.forEach((name, idx) => {
      const optX = document.createElement("option");
      optX.value = idx;
      optX.textContent = name;
      if (idx === axX) optX.selected = true;
      selX.appendChild(optX);

      const optY = document.createElement("option");
      optY.value = idx;
      optY.textContent = name;
      if (idx === axY) optY.selected = true;
      selY.appendChild(optY);
    });
  }

  function renderPreviewTable(featureNames, targetName, rows) {
    const headerRow = document.getElementById("previewHeaderRow");
    const bodyRows = document.getElementById("previewBodyRows");
    headerRow.innerHTML = "";
    bodyRows.innerHTML = "";

    const extraCols = rows.length && rows[0].Pred_PD_Pct !== undefined
      ? ["Pred_PD_Pct", "Credit_Score", "Rating"]
      : [];
    const cols = [...featureNames, ...extraCols, targetName];

    cols.forEach((c) => {
      const th = document.createElement("th");
      th.className = "px-3 py-2.5 font-semibold whitespace-nowrap text-emerald-400 border-b border-slate-800";
      th.textContent = c;
      headerRow.appendChild(th);
    });

    rows.forEach((r) => {
      const tr = document.createElement("tr");
      tr.className = "hover:bg-slate-800/60 transition-colors";
      cols.forEach((c) => {
        const td = document.createElement("td");
        td.className = "px-3 py-2 text-slate-300 whitespace-nowrap";
        if (c === targetName) {
          const isDef = Number(r[c]) === 1;
          td.innerHTML = isDef
            ? `<span class="px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 border border-rose-500/40 font-semibold">1 · DEFAULT</span>`
            : `<span class="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-semibold">0 · PERFORMING</span>`;
        } else if (c === "Rating") {
          td.innerHTML = `<span class="px-2 py-0.5 rounded bg-amber-500/15 text-amber-300 border border-amber-500/30 font-bold">${r[c]}</span>`;
        } else {
          td.textContent = r[c] !== undefined ? r[c] : "--";
        }
        tr.appendChild(td);
      });
      bodyRows.appendChild(tr);
    });
  }

  function renderInferenceSliders(featureRanges) {
    const container = document.getElementById("inferenceSliders");
    container.innerHTML = "";

    featureRanges.forEach((fr) => {
      const step = ((fr.max - fr.min) / 100) || 0.05;
      const ivText = fr.iv !== undefined ? `IV ${fr.iv.toFixed(2)}` : "";
      const div = document.createElement("div");
      div.className = "bg-slate-900/90 p-2.5 rounded-lg border border-slate-800 hover:border-emerald-500/40 transition";
      div.innerHTML = `
        <div class="flex justify-between items-center text-xs mb-1.5">
          <span class="font-medium text-slate-300">${fr.name} <span class="text-[10px] text-amber-400/90 font-mono ml-1 px-1.5 py-0.5 rounded bg-amber-500/10">${ivText}</span></span>
          <span id="lblInf_${fr.index}" class="font-mono font-bold text-emerald-400">${fr.mean.toFixed(2)}</span>
        </div>
        <input type="range" data-idx="${fr.index}" class="infSlider w-full accent-emerald-500 cursor-pointer" min="${fr.min}" max="${fr.max}" step="${step.toFixed(4)}" value="${fr.mean}">
      `;
      container.appendChild(div);
    });

    document.querySelectorAll(".infSlider").forEach((sl) => {
      sl.addEventListener("input", (e) => {
        const idx = e.target.getAttribute("data-idx");
        const lbl = document.getElementById(`lblInf_${idx}`);
        if (lbl) lbl.textContent = parseFloat(e.target.value).toFixed(2);
        updateLivePrediction();
      });
    });

    updateLivePrediction();
  }

  async function updateLivePrediction() {
    const sliders = Array.from(document.querySelectorAll(".infSlider"));
    if (!sliders.length) return;
    const features = sliders.map((s) => parseFloat(s.value));
    const cfg = getConfigFromUI();

    const xVal = features[state.axisX] !== undefined ? features[state.axisX] : features[0];
    const yVal = features[state.axisY] !== undefined ? features[state.axisY] : features[Math.min(1, features.length - 1)];
    window.Visualizer.updateBorrowerMarker(xVal, yVal);

    if (state.backendAvailable) {
      try {
        const json = await postBackendAPI("/api/predict", {
          features,
          loan_amount_k: cfg.eadK,
          lgd_pct: cfg.lgdPct
        });
        if (json.status === "ok") {
          applyPredictionToUI(json.prediction);
          return;
        }
      } catch (_) {}
    }

    if (state.activePredictor) {
      const pred = state.activePredictor(features, cfg.eadK, cfg.lgdPct);
      applyPredictionToUI(pred);
    }
  }

  function applyPredictionToUI(p) {
    const badge = document.getElementById("predBadge");
    badge.textContent = p.decision;
    if (p.decision.startsWith("APPROVE")) {
      badge.className = "px-3 py-1 rounded-md text-xs font-mono font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/50";
    } else if (p.decision.startsWith("CONDITIONAL")) {
      badge.className = "px-3 py-1 rounded-md text-xs font-mono font-bold bg-amber-500/20 text-amber-300 border border-amber-500/50";
    } else {
      badge.className = "px-3 py-1 rounded-md text-xs font-mono font-bold bg-rose-500/20 text-rose-300 border border-rose-500/50";
    }

    const pPerf = (p.prob_performing * 100).toFixed(1);
    const pDef = (p.prob_default * 100).toFixed(1);
    document.getElementById("predProb0").textContent = `Performing: ${pPerf}%`;
    document.getElementById("predProb1").textContent = `Default PD: ${pDef}%`;
    document.getElementById("barProb0").style.width = `${pPerf}%`;
    document.getElementById("barProb1").style.width = `${pDef}%`;

    document.getElementById("borrowerScore").textContent = `${p.credit_score} (${p.rating_grade})`;
    document.getElementById("borrowerStage").textContent = p.ifrs9_stage;
    document.getElementById("borrowerECL").textContent = `$${p.basel.ecl.toFixed(2)}k`;
    document.getElementById("borrowerRWA").textContent = `$${p.basel.rwa.toFixed(1)}k (${p.basel.risk_weight_pct}%)`;
    document.getElementById("borrowerCap").textContent = `$${p.basel.regulatory_capital_8pct.toFixed(2)}k`;
    document.getElementById("borrowerAPR").textContent = `${p.recommended_apr_pct.toFixed(2)}% (+${p.risk_spread_bps} bps · RAROC ${p.raroc_pct}%)`;
  }

  function exportActiveCSV() {
    if (!state.dataset) return;
    const { rows, featureNames, targetName } = state.dataset;
    const header = [...featureNames, targetName].join(",");
    const lines = rows.map((r) => [...r.X.map((v) => Number(v.toFixed(4))), r.y].join(","));
    const csv = [header, ...lines].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `credit_portfolio_${state.dataset.preset || "export"}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  window.addEventListener("DOMContentLoaded", () => {
    const tabSyn = document.getElementById("tabBtnSynthetic");
    const tabCSV = document.getElementById("tabBtnCSV");
    const panelSyn = document.getElementById("panelSynthetic");
    const panelCSV = document.getElementById("panelCSV");

    tabSyn.addEventListener("click", () => {
      tabSyn.className = "px-3 py-1 rounded bg-emerald-500 text-slate-950 font-semibold shadow-sm";
      tabCSV.className = "px-3 py-1 rounded text-slate-400 hover:text-white";
      panelSyn.classList.remove("hidden");
      panelCSV.classList.add("hidden");
    });

    tabCSV.addEventListener("click", () => {
      tabCSV.className = "px-3 py-1 rounded bg-emerald-500 text-slate-950 font-semibold shadow-sm";
      tabSyn.className = "px-3 py-1 rounded text-slate-400 hover:text-white";
      panelCSV.classList.remove("hidden");
      panelSyn.classList.add("hidden");
    });

    // Workspace View Mode Switcher
    const zones = {
      surface: document.getElementById("zoneSurfaceRow"),
      diagnostics: document.getElementById("zoneUnderwritingDiagnosticsRow"),
      blotter: document.getElementById("zoneBlotterRow")
    };
    document.querySelectorAll(".workspaceModeBtn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const mode = btn.getAttribute("data-mode");
        document.querySelectorAll(".workspaceModeBtn").forEach((b) => {
          b.className = "workspaceModeBtn px-3 py-1.5 rounded-md text-xs font-medium text-slate-400 hover:text-white hover:bg-slate-800/80 transition";
        });
        btn.className = "workspaceModeBtn px-3 py-1.5 rounded-md text-xs font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40";

        if (mode === "all") {
          Object.values(zones).forEach((z) => z && z.classList.remove("hidden"));
        } else {
          Object.keys(zones).forEach((k) => {
            if (zones[k]) zones[k].classList.toggle("hidden", k !== mode);
          });
        }
      });
    });

    document.getElementById("inputSamples").addEventListener("input", (e) => {
      document.getElementById("valSamples").textContent = e.target.value;
    });
    document.getElementById("inputNoise").addEventListener("input", (e) => {
      document.getElementById("valNoise").textContent = parseFloat(e.target.value).toFixed(2);
    });
    document.getElementById("inputLR").addEventListener("input", (e) => {
      document.getElementById("valLR").textContent = parseFloat(e.target.value).toFixed(2);
    });
    document.getElementById("inputEpochs").addEventListener("input", (e) => {
      document.getElementById("valEpochs").textContent = e.target.value;
    });
    document.getElementById("inputHidden").addEventListener("input", (e) => {
      document.getElementById("valHidden").textContent = e.target.value;
    });
    document.getElementById("inputDepth").addEventListener("input", (e) => {
      document.getElementById("valDepth").textContent = e.target.value;
    });
    document.getElementById("inputThreshold").addEventListener("input", (e) => {
      document.getElementById("valThreshold").textContent = `${e.target.value}%`;
      runTraining();
    });
    document.getElementById("inputLGD").addEventListener("input", (e) => {
      document.getElementById("valLGD").textContent = `${e.target.value}%`;
      runTraining();
    });
    document.getElementById("inputEAD").addEventListener("input", (e) => {
      document.getElementById("valEAD").textContent = `$${e.target.value}k`;
      runTraining();
    });

    document.getElementById("selectPreset").addEventListener("change", generateNewCreditPortfolio);
    document.getElementById("btnGenerateData").addEventListener("click", generateNewCreditPortfolio);
    document.getElementById("btnExportCSV").addEventListener("click", exportActiveCSV);
    document.getElementById("btnTrainModel").addEventListener("click", runTraining);

    document.getElementById("btnOptimalKSCutoff").addEventListener("click", () => {
      if (!state.lastData || !state.lastData.metrics) return;
      const optPct = Math.round(state.lastData.metrics.ks_optimal_threshold * 100);
      const thSlider = document.getElementById("inputThreshold");
      thSlider.value = Math.max(10, Math.min(75, optPct));
      document.getElementById("valThreshold").textContent = `${thSlider.value}%`;
      runTraining();
    });

    document.getElementById("selectModelType").addEventListener("change", () => {
      runTraining();
    });

    document.getElementById("fileInputCSV").addEventListener("change", async (e) => {
      if (!e.target.files || !e.target.files[0]) return;
      const text = await e.target.files[0].text();
      try {
        const { headers } = window.DataGenerator.parseRawCSV(text);
        state.rawCSVText = text;
        const selTarget = document.getElementById("selectTargetCol");
        selTarget.innerHTML = "";
        headers.forEach((h, i) => {
          const opt = document.createElement("option");
          opt.value = h;
          opt.textContent = h;
          if (i === headers.length - 1) opt.selected = true;
          selTarget.appendChild(opt);
        });
        document.getElementById("csvColumnSelector").classList.remove("hidden");
        state.dataset = window.DataGenerator.parseCSVToDataset(text, headers[headers.length - 1]);
        state.axisX = 0;
        state.axisY = Math.min(1, state.dataset.featureNames.length - 1);
        await runTraining();
      } catch (err) {
        setStatus(`CSV Error: ${err.message}`, true);
      }
    });

    document.getElementById("btnApplyUploadedCSV").addEventListener("click", async () => {
      if (!state.rawCSVText) return;
      const targetCol = document.getElementById("selectTargetCol").value;
      state.dataset = window.DataGenerator.parseCSVToDataset(state.rawCSVText, targetCol);
      state.axisX = 0;
      state.axisY = Math.min(1, state.dataset.featureNames.length - 1);
      await runTraining();
    });

    document.querySelectorAll(".btnSampleCSV").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const key = btn.getAttribute("data-sample");
        state.dataset = window.DataGenerator.getSampleDataset(key);
        state.axisX = 0;
        state.axisY = Math.min(1, state.dataset.featureNames.length - 1);
        await runTraining();
      });
    });

    document.getElementById("selectAxisX").addEventListener("change", (e) => {
      state.axisX = parseInt(e.target.value, 10);
      runTraining();
    });
    document.getElementById("selectAxisY").addEventListener("change", (e) => {
      state.axisY = parseInt(e.target.value, 10);
      runTraining();
    });

    const views = {
      ks: document.getElementById("viewChartKS"),
      decile: document.getElementById("viewChartDecile"),
      roc: document.getElementById("viewChartROC"),
      weights: document.getElementById("viewChartWeights"),
      cm: document.getElementById("viewChartCM"),
      loss: document.getElementById("viewChartLoss")
    };

    document.querySelectorAll(".tabChartBtn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const key = btn.getAttribute("data-chart");
        document.querySelectorAll(".tabChartBtn").forEach((b) => {
          b.className = "tabChartBtn px-3 py-1.5 rounded-md text-xs text-slate-400 hover:bg-slate-800 hover:text-white transition";
        });
        btn.className = "tabChartBtn px-3 py-1.5 rounded-md text-xs bg-emerald-500 text-slate-950 font-bold shadow-sm";
        Object.keys(views).forEach((k) => views[k].classList.toggle("hidden", k !== key));
      });
    });

    if (window.location.protocol.startsWith("http") && window.location.port === "5000") {
      state.backendAvailable = true;
    }
    generateNewCreditPortfolio();
  });
})();

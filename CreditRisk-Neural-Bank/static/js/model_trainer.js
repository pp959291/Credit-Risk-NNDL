/**
 * static/js/model_trainer.js
 * Specialized Credit Risk & Basel III Banking Machine Learning Engine.
 * Implements:
 * 1. WoE & IV Bank Credit Scorecard (PDO 300-850 Scaling)
 * 2. Cost-Sensitive Credit Risk Neural Network (2-Layer MLP with LeakyReLU & Momentum)
 * 3. Non-Linear Leverage & Stress Interaction Network
 * 4. Gradient Boosted Credit Decision Trees (GBDT)
 * 5. Basel III IRB RWA, Regulatory Capital, Expected Credit Loss (ECL), IFRS 9 Stage, KS, Gini, and Lift Analytics
 */
window.ModelTrainer = (function () {
  function sigmoid(z) {
    const zc = Math.max(-250, Math.min(250, z));
    return 1.0 / (1.0 + Math.exp(-zc));
  }

  function erfApprox(x) {
    const sign = x >= 0 ? 1.0 : -1.0;
    const ax = Math.abs(x);
    const t = 1.0 / (1.0 + 0.3275911 * ax);
    const y = 1.0 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t) * Math.exp(-ax * ax);
    return sign * y;
  }

  function normCdf(x) {
    return 0.5 * (1.0 + erfApprox(x / Math.SQRT2));
  }

  function normPpf(pVal) {
    const p = Math.max(1e-6, Math.min(1.0 - 1e-6, pVal));
    const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
    const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
    const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
    const plow = 0.02425;
    const phigh = 1.0 - plow;
    if (p < plow) {
      const q = Math.sqrt(-2.0 * Math.log(p));
      return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
             ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if (phigh < p) {
      const q = Math.sqrt(-2.0 * Math.log(1.0 - p));
      return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    const q = p - 0.5;
    const r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
  }

  function computeBaselIRBCapital(pdVal, lgdVal, eadVal) {
    const pd = Math.max(0.0003, Math.min(0.999, pdVal));
    const lgd = Math.max(0.05, Math.min(1.0, lgdVal));
    const ead = Math.max(0.0, eadVal);

    const expFactor = (1.0 - Math.exp(-50.0 * pd)) / (1.0 - Math.exp(-50.0));
    const rCorr = 0.12 * expFactor + 0.24 * (1.0 - expFactor);
    const bMat = Math.pow(0.11852 - 0.05478 * Math.log(pd), 2);
    const maturityAdj = 1.0 / Math.max(0.25, 1.0 - 1.5 * bMat);

    const invPd = normPpf(pd);
    const inv999 = normPpf(0.999);
    const condPd = normCdf((invPd + Math.sqrt(rCorr) * inv999) / Math.sqrt(1.0 - rCorr));

    const kFactor = Math.max(0.0, (lgd * condPd - pd * lgd) * maturityAdj);
    const rwa = 12.5 * kFactor * ead;
    const regCap8 = 0.08 * rwa;
    const totCap105 = 0.105 * rwa;
    const ecl = pd * lgd * ead;
    const rwPct = ead > 0 ? (rwa / ead) * 100.0 : 0.0;

    return {
      pd: Number(pd.toFixed(5)),
      lgd: Number(lgd.toFixed(4)),
      ead: Number(ead.toFixed(2)),
      ecl: Number(ecl.toFixed(3)),
      asset_correlation_r: Number(rCorr.toFixed(4)),
      capital_factor_k: Number(kFactor.toFixed(4)),
      rwa: Number(rwa.toFixed(2)),
      risk_weight_pct: Number(rwPct.toFixed(1)),
      regulatory_capital_8pct: Number(regCap8.toFixed(3)),
      total_capital_10_5pct: Number(totCap105.toFixed(3))
    };
  }

  function pdToCreditScoreAndGrade(pdVal) {
    const p = Math.max(1e-4, Math.min(1.0 - 1e-4, pdVal));
    const odds = (1.0 - p) / p;
    const factor = 50.0 / Math.LN2;
    const offset = 650.0 - factor * Math.log(19.0);
    const score = Math.round(Math.max(300, Math.min(850, offset + factor * Math.log(odds))));

    let grade = "CCC/D";
    let stage = "Stage 3 (Impaired / High Risk)";
    if (p < 0.015) { grade = "AAA"; stage = "Stage 1 (Performing)"; }
    else if (p < 0.030) { grade = "AA"; stage = "Stage 1 (Performing)"; }
    else if (p < 0.055) { grade = "A"; stage = "Stage 1 (Performing)"; }
    else if (p < 0.100) { grade = "BBB"; stage = "Stage 2 (SICR Watch)"; }
    else if (p < 0.180) { grade = "BB"; stage = "Stage 2 (SICR Watch)"; }
    else if (p < 0.300) { grade = "B"; stage = "Stage 2 (Elevated Risk)"; }

    return { score, grade, stage };
  }

  function computeWoEAndIV(X, y, nBins = 5) {
    const n = X.length;
    const nFeats = X[0].length;
    let totalBad = 0;
    for (let i = 0; i < n; i++) if (y[i] === 1) totalBad++;
    const totalGood = Math.max(1, n - totalBad);
    totalBad = Math.max(1, totalBad);

    const ivScores = [];
    const binModels = [];

    for (let j = 0; j < nFeats; j++) {
      const col = X.map((r) => r[j]).slice().sort((a, b) => a - b);
      const edges = [];
      for (let b = 0; b <= nBins; b++) {
        const idx = Math.min(col.length - 1, Math.floor((b / nBins) * col.length));
        edges.push(col[idx]);
      }
      edges[0] -= 1e-5;
      edges[edges.length - 1] += 1e-5;

      const woeArr = [];
      let iv = 0.0;
      for (let b = 0; b < nBins; b++) {
        const low = edges[b], high = edges[b + 1];
        let bBad = 0.5, bGood = 0.5;
        for (let i = 0; i < n; i++) {
          const v = X[i][j];
          const inBin = (b === nBins - 1) ? (v >= low && v <= high) : (v >= low && v < high);
          if (inBin) {
            if (y[i] === 1) bBad += 1.0;
            else bGood += 1.0;
          }
        }
        const distBad = bBad / (totalBad + 0.5 * nBins);
        const distGood = bGood / (totalGood + 0.5 * nBins);
        const woe = Math.log(distBad / distGood);
        iv += (distBad - distGood) * woe;
        woeArr.push(woe);
      }
      ivScores.push(Number(Math.abs(iv).toFixed(4)));
      binModels.push({ edges, woeArr });
    }
    return { ivScores, binModels };
  }

  class WoEScorecardModel {
    constructor(lr = 0.15, epochs = 200, costWeight = 1.6) {
      this.lr = lr;
      this.epochs = epochs;
      this.costWeight = costWeight;
      this.binModels = [];
      this.ivScores = [];
      this.weights = [];
      this.bias = 0.0;
      this.lossHistory = [];
    }

    transformRow(x) {
      return x.map((v, j) => {
        const bm = this.binModels[j];
        let binIdx = bm.woeArr.length - 1;
        for (let b = 0; b < bm.woeArr.length; b++) {
          if (v >= bm.edges[b] && v < bm.edges[b + 1]) {
            binIdx = b;
            break;
          }
        }
        return 0.65 * v + 0.35 * bm.woeArr[binIdx];
      });
    }

    fit(X, y) {
      const woeInfo = computeWoEAndIV(X, y, 5);
      this.ivScores = woeInfo.ivScores;
      this.binModels = woeInfo.binModels;

      const X_des = X.map((r) => this.transformRow(r));
      const m = X_des.length;
      const n = X_des[0].length;
      this.weights = new Array(n).fill(0).map((_, idx) => Math.sin(idx + 1) * 0.05);
      this.bias = 0.0;
      this.lossHistory = [];
      const l2 = 0.002;
      const eps = 1e-15;

      for (let ep = 1; ep <= this.epochs; ep++) {
        let loss = 0.0;
        const dw = new Array(n).fill(0.0);
        let db = 0.0;

        for (let i = 0; i < m; i++) {
          let z = this.bias;
          for (let j = 0; j < n; j++) z += X_des[i][j] * this.weights[j];
          const p = sigmoid(z);
          const pClip = Math.max(eps, Math.min(1.0 - eps, p));
          const sw = y[i] === 1 ? this.costWeight : 1.0;
          loss += -sw * (y[i] * Math.log(pClip) + (1 - y[i]) * Math.log(1 - pClip));

          const err = sw * (p - y[i]);
          for (let j = 0; j < n; j++) dw[j] += err * X_des[i][j];
          db += err;
        }

        for (let j = 0; j < n; j++) {
          this.weights[j] -= this.lr * ((dw[j] / m) + (l2 / m) * this.weights[j]);
        }
        this.bias -= this.lr * (db / m);
        this.lossHistory.push({ epoch: ep, loss: Number((loss / m).toFixed(5)) });
      }
    }

    predictProb(x) {
      const xd = this.transformRow(x);
      let z = this.bias;
      for (let j = 0; j < xd.length; j++) z += xd[j] * this.weights[j];
      return sigmoid(z);
    }

    getFeatureWeights() {
      return this.weights.map((w) => Number(w.toFixed(4)));
    }
  }

  class CreditRiskNeuralMLPModel {
    constructor(hiddenDim = 14, lr = 0.12, epochs = 200, costWeight = 1.65) {
      this.hiddenDim = Math.max(4, hiddenDim);
      this.lr = lr;
      this.epochs = epochs;
      this.costWeight = costWeight;
      this.W1 = [];
      this.b1 = [];
      this.W2 = [];
      this.b2 = 0.0;
      this.effWeights = [];
      this.lossHistory = [];
    }

    fit(X, y) {
      const m = X.length;
      const n = X[0].length;
      const H = this.hiddenDim;

      this.W1 = Array.from({ length: n }, (_, i) =>
        Array.from({ length: H }, (__, h) => Math.sin((i + 1) * (h + 2)) * Math.sqrt(1.5 / n))
      );
      this.b1 = new Array(H).fill(0.0);
      this.W2 = Array.from({ length: H }, (_, h) => Math.cos(h + 1) * Math.sqrt(1.0 / H));
      this.b2 = 0.0;

      const vW1 = Array.from({ length: n }, () => new Array(H).fill(0.0));
      const vb1 = new Array(H).fill(0.0);
      const vW2 = new Array(H).fill(0.0);
      let vb2 = 0.0;
      const mom = 0.82;
      const l2 = 0.0015;
      const eps = 1e-15;
      this.lossHistory = [];

      for (let ep = 1; ep <= this.epochs; ep++) {
        let loss = 0.0;
        const dW1 = Array.from({ length: n }, () => new Array(H).fill(0.0));
        const db1 = new Array(H).fill(0.0);
        const dW2 = new Array(H).fill(0.0);
        let db2 = 0.0;

        for (let i = 0; i < m; i++) {
          const z1 = new Array(H);
          const a1 = new Array(H);
          let z2 = this.b2;

          for (let h = 0; h < H; h++) {
            let s = this.b1[h];
            for (let j = 0; j < n; j++) s += X[i][j] * this.W1[j][h];
            z1[h] = s;
            a1[h] = s > 0 ? s : 0.1 * s;
            z2 += a1[h] * this.W2[h];
          }

          const p = sigmoid(z2);
          const pClip = Math.max(eps, Math.min(1.0 - eps, p));
          const sw = y[i] === 1 ? this.costWeight : 1.0;
          loss += -sw * (y[i] * Math.log(pClip) + (1 - y[i]) * Math.log(1 - pClip));

          const dz2 = sw * (p - y[i]) / m;
          db2 += dz2;
          for (let h = 0; h < H; h++) {
            dW2[h] += a1[h] * dz2;
            const da1 = dz2 * this.W2[h];
            const dz1 = da1 * (z1[h] > 0 ? 1.0 : 0.1);
            db1[h] += dz1;
            for (let j = 0; j < n; j++) {
              dW1[j][h] += X[i][j] * dz1;
            }
          }
        }

        for (let h = 0; h < H; h++) {
          vW2[h] = mom * vW2[h] - this.lr * (dW2[h] + (l2 / m) * this.W2[h]);
          this.W2[h] += vW2[h];
          vb1[h] = mom * vb1[h] - this.lr * db1[h];
          this.b1[h] += vb1[h];
          for (let j = 0; j < n; j++) {
            vW1[j][h] = mom * vW1[j][h] - this.lr * (dW1[j][h] + (l2 / m) * this.W1[j][h]);
            this.W1[j][h] += vW1[j][h];
          }
        }
        vb2 = mom * vb2 - this.lr * db2;
        this.b2 += vb2;

        this.lossHistory.push({ epoch: ep, loss: Number((loss / m).toFixed(5)) });
      }

      this.effWeights = new Array(n).fill(0.0);
      for (let j = 0; j < n; j++) {
        let sum = 0.0;
        for (let h = 0; h < H; h++) sum += this.W1[j][h] * this.W2[h];
        this.effWeights[j] = Number(sum.toFixed(4));
      }
    }

    predictProb(x) {
      const H = this.hiddenDim;
      let z2 = this.b2;
      for (let h = 0; h < H; h++) {
        let s = this.b1[h];
        for (let j = 0; j < x.length; j++) s += x[j] * this.W1[j][h];
        const a1 = s > 0 ? s : 0.1 * s;
        z2 += a1 * this.W2[h];
      }
      return sigmoid(z2);
    }

    getFeatureWeights() {
      return this.effWeights;
    }
  }

  function expandPolyFeatures(x) {
    const res = [...x];
    for (let i = 0; i < x.length; i++) {
      res.push(Math.max(-6, Math.min(6, x[i] * x[i])));
    }
    for (let i = 0; i < x.length; i++) {
      for (let j = i + 1; j < x.length; j++) {
        res.push(Math.max(-6, Math.min(6, x[i] * x[j])));
      }
    }
    return res;
  }

  class CreditInteractionModel {
    constructor(lr = 0.14, epochs = 200, costWeight = 1.6) {
      this.lr = lr;
      this.epochs = epochs;
      this.costWeight = costWeight;
      this.weights = [];
      this.bias = 0.0;
      this.rawDim = 0;
      this.lossHistory = [];
    }

    fit(X, y) {
      this.rawDim = X[0].length;
      const X_mat = X.map(expandPolyFeatures);
      const m = X_mat.length;
      const n = X_mat[0].length;

      this.weights = new Array(n).fill(0).map((_, idx) => Math.sin(idx + 1) * 0.04);
      this.bias = 0.0;
      this.lossHistory = [];
      const l2 = 0.002;
      const eps = 1e-15;

      for (let ep = 1; ep <= this.epochs; ep++) {
        let loss = 0.0;
        const dw = new Array(n).fill(0.0);
        let db = 0.0;

        for (let i = 0; i < m; i++) {
          let z = this.bias;
          for (let j = 0; j < n; j++) z += X_mat[i][j] * this.weights[j];
          const p = sigmoid(z);
          const pClip = Math.max(eps, Math.min(1.0 - eps, p));
          const sw = y[i] === 1 ? this.costWeight : 1.0;
          loss += -sw * (y[i] * Math.log(pClip) + (1 - y[i]) * Math.log(1 - pClip));

          const err = sw * (p - y[i]);
          for (let j = 0; j < n; j++) dw[j] += err * X_mat[i][j];
          db += err;
        }

        for (let j = 0; j < n; j++) {
          this.weights[j] -= this.lr * ((dw[j] / m) + (l2 / m) * this.weights[j]);
        }
        this.bias -= this.lr * (db / m);
        this.lossHistory.push({ epoch: ep, loss: Number((loss / m).toFixed(5)) });
      }
    }

    predictProb(x) {
      const vec = expandPolyFeatures(x);
      let z = this.bias;
      for (let j = 0; j < vec.length; j++) z += vec[j] * this.weights[j];
      return sigmoid(z);
    }

    getFeatureWeights() {
      return this.weights.slice(0, this.rawDim).map((w) => Number(w.toFixed(4)));
    }
  }

  class CreditGBDTModel {
    constructor(nTrees = 22, maxDepth = 3, lr = 0.22) {
      this.nTrees = Math.max(5, nTrees);
      this.maxDepth = Math.max(1, maxDepth);
      this.lr = lr;
      this.trees = [];
      this.initLogOdds = 0.0;
      this.importances = [];
      this.lossHistory = [];
    }

    variance(arr) {
      if (!arr.length) return 0;
      const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
      let s = 0;
      for (let i = 0; i < arr.length; i++) s += (arr[i] - mean) * (arr[i] - mean);
      return s;
    }

    buildTree(X, res, hess, depth) {
      const n = res.length;
      const sumRes = res.reduce((a, b) => a + b, 0);
      const sumHess = hess.reduce((a, b) => a + b, 0) + 1e-3;
      if (depth >= this.maxDepth || n <= 6) {
        return { leaf: true, val: sumRes / sumHess };
      }

      const baseVar = this.variance(res);
      if (baseVar < 1e-6) return { leaf: true, val: sumRes / sumHess };

      let bestGain = 0.0;
      let bestSplit = null;
      const nFeats = X[0].length;

      for (let f = 0; f < nFeats; f++) {
        const vals = X.map((r) => r[f]).slice().sort((a, b) => a - b);
        const step = Math.max(1, Math.floor(vals.length / 10));
        for (let i = step; i < vals.length; i += step) {
          const th = (vals[i - 1] + vals[i]) * 0.5;
          const lX = [], lR = [], lH = [], rX = [], rR = [], rH = [];
          for (let r = 0; r < n; r++) {
            if (X[r][f] <= th) { lX.push(X[r]); lR.push(res[r]); lH.push(hess[r]); }
            else { rX.push(X[r]); rR.push(res[r]); rH.push(hess[r]); }
          }
          if (lR.length < 3 || rR.length < 3) continue;
          const gain = baseVar - (this.variance(lR) + this.variance(rR));
          if (gain > bestGain) {
            bestGain = gain;
            bestSplit = { f, th, lX, lR, lH, rX, rR, rH };
          }
        }
      }

      if (!bestSplit) return { leaf: true, val: sumRes / sumHess };
      this.importances[bestSplit.f] += bestGain;
      return {
        leaf: false,
        feature: bestSplit.f,
        threshold: bestSplit.th,
        left: this.buildTree(bestSplit.lX, bestSplit.lR, bestSplit.lH, depth + 1),
        right: this.buildTree(bestSplit.rX, bestSplit.rR, bestSplit.rH, depth + 1)
      };
    }

    traverse(node, x) {
      if (node.leaf) return node.val;
      return x[node.feature] <= node.threshold ? this.traverse(node.left, x) : this.traverse(node.right, x);
    }

    fit(X, y) {
      const m = X.length;
      const nFeats = X[0].length;
      this.importances = new Array(nFeats).fill(0);
      const pBase = Math.max(0.05, Math.min(0.95, y.reduce((a, b) => a + b, 0) / m));
      this.initLogOdds = Math.log(pBase / (1.0 - pBase));
      const fPred = new Array(m).fill(this.initLogOdds);
      this.trees = [];
      this.lossHistory = [];
      const eps = 1e-15;

      for (let t = 1; t <= this.nTrees; t++) {
        const res = new Array(m);
        const hess = new Array(m);
        for (let i = 0; i < m; i++) {
          const p = sigmoid(fPred[i]);
          res[i] = y[i] - p;
          hess[i] = Math.max(1e-4, p * (1.0 - p));
        }
        const tree = this.buildTree(X, res, hess, 0);
        this.trees.push(tree);

        let loss = 0.0;
        for (let i = 0; i < m; i++) {
          fPred[i] += this.lr * this.traverse(tree, X[i]);
          const pNew = Math.max(eps, Math.min(1.0 - eps, sigmoid(fPred[i])));
          loss += -(y[i] * Math.log(pNew) + (1 - y[i]) * Math.log(1 - pNew));
        }
        this.lossHistory.push({ epoch: t, loss: Number((loss / m).toFixed(5)) });
      }

      const sumImp = this.importances.reduce((a, b) => a + b, 0) || 1;
      this.importances = this.importances.map((v) => Number((v / sumImp).toFixed(4)));
    }

    predictProb(x) {
      let f = this.initLogOdds;
      for (let t = 0; t < this.trees.length; t++) {
        f += this.lr * this.traverse(this.trees[t], x);
      }
      return sigmoid(f);
    }

    getFeatureWeights() {
      return this.importances;
    }
  }

  function trainPipeline(dataset, config) {
    const { rows, featureNames, targetName, classNames, meta = {} } = dataset;
    const {
      modelType = "woe_scorecard",
      lr = 0.14,
      epochs = 200,
      hiddenDim = 14,
      treeDepth = 3,
      threshold = 0.35,
      lgdPct = meta.defaultLGD || 65,
      eadK = meta.avgEAD || 18.5,
      baseRatePct = meta.baseRate || 6.5,
      standardize = true,
      axisX = 0,
      axisY = 1
    } = config;

    const n = rows.length;
    const nFeats = featureNames.length;
    const lgdVal = lgdPct / 100.0;

    const means = new Array(nFeats).fill(0);
    const stds = new Array(nFeats).fill(0);
    const mins = new Array(nFeats).fill(Infinity);
    const maxs = new Array(nFeats).fill(-Infinity);

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < nFeats; j++) {
        const val = rows[i].X[j];
        means[j] += val / n;
        if (val < mins[j]) mins[j] = val;
        if (val > maxs[j]) maxs[j] = val;
      }
    }

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < nFeats; j++) {
        stds[j] += Math.pow(rows[i].X[j] - means[j], 2) / n;
      }
    }
    for (let j = 0; j < nFeats; j++) {
      stds[j] = Math.sqrt(stds[j]) || 1.0;
    }

    function scaleVec(vec) {
      if (!standardize) return [...vec];
      return vec.map((v, j) => (v - means[j]) / stds[j]);
    }

    const splitIdx = Math.floor(n * 0.8);
    const trainRows = rows.slice(0, splitIdx);
    const testRows = rows.slice(splitIdx);

    const X_train = trainRows.map((r) => scaleVec(r.X));
    const y_train = trainRows.map((r) => r.y);
    const X_test = testRows.map((r) => scaleVec(r.X));
    const y_test = testRows.map((r) => r.y);
    const X_all = rows.map((r) => scaleVec(r.X));
    const y_all = rows.map((r) => r.y);

    const woeSummary = computeWoEAndIV(X_train, y_train, 5);
    const ivScores = woeSummary.ivScores;

    let model;
    if (modelType === "woe_scorecard" || modelType === "logistic") {
      model = new WoEScorecardModel(lr, epochs, 1.6);
    } else if (modelType === "credit_neural_mlp") {
      model = new CreditRiskNeuralMLPModel(hiddenDim, lr, epochs, 1.65);
    } else if (modelType === "credit_interaction_net" || modelType === "poly_logistic") {
      model = new CreditInteractionModel(lr, epochs, 1.6);
    } else {
      model = new CreditGBDTModel(Math.min(40, Math.max(12, Math.floor(epochs / 6))), treeDepth, Math.min(0.35, lr * 1.5));
    }

    model.fit(X_train, y_train);

    let trainCorrect = 0;
    for (let i = 0; i < X_train.length; i++) {
      const pred = model.predictProb(X_train[i]) >= threshold ? 1 : 0;
      if (pred === y_train[i]) trainCorrect++;
    }
    const trainAcc = trainCorrect / (X_train.length || 1);

    let tp = 0, tn = 0, fp = 0, fn = 0;
    let brierSum = 0.0;
    const testScored = [];
    for (let i = 0; i < X_test.length; i++) {
      const prob = model.predictProb(X_test[i]);
      const pred = prob >= threshold ? 1 : 0;
      const actual = y_test[i];
      brierSum += Math.pow(prob - actual, 2);
      testScored.push({ prob, actual });
      if (pred === 1 && actual === 1) tp++;
      else if (pred === 0 && actual === 0) tn++;
      else if (pred === 1 && actual === 0) fp++;
      else fn++;
    }

    const testAcc = (tp + tn) / (X_test.length || 1);
    const brierScore = brierSum / (X_test.length || 1);
    const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
    const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
    const f1 = (precision + recall) > 0 ? (2 * precision * recall) / (precision + recall) : 0;

    // ROC, Gini, and KS Curve
    const totalPos = tp + fn || 1;
    const totalNeg = tn + fp || 1;
    const rocCurve = [];
    const ksCurve = [];
    const fprArr = [];
    const tprArr = [];
    let maxKS = 0.0;
    let ksOptTh = 0.35;

    for (let step = 50; step >= 0; step--) {
      const th = step / 50.0;
      let cTP = 0, cFP = 0;
      for (let i = 0; i < testScored.length; i++) {
        if (testScored[i].prob >= th) {
          if (testScored[i].actual === 1) cTP++;
          else cFP++;
        }
      }
      const fpr = cFP / totalNeg;
      const tpr = cTP / totalPos;
      const ksDiff = Math.abs(tpr - fpr);
      if (ksDiff > maxKS) {
        maxKS = ksDiff;
        ksOptTh = th;
      }
      fprArr.push(fpr);
      tprArr.push(tpr);
      rocCurve.push({ x: Number(fpr.toFixed(4)), y: Number(tpr.toFixed(4)) });
      ksCurve.unshift({
        threshold: Number(th.toFixed(2)),
        cum_bad: Number((tpr * 100).toFixed(2)),
        cum_good: Number((fpr * 100).toFixed(2)),
        ks_gap: Number((ksDiff * 100).toFixed(2))
      });
    }

    let auc = 0.0;
    for (let i = 1; i < fprArr.length; i++) {
      auc += (fprArr[i] - fprArr[i - 1]) * (tprArr[i] + tprArr[i - 1]) * 0.5;
    }
    auc = Math.max(0.5, Math.min(1.0, Math.abs(auc)));
    const giniCoeff = Math.max(0.0, 2.0 * auc - 1.0);

    // Full Portfolio Probabilities, Decile Lift, and Basel III Summary
    const allScored = X_all.map((xVec, idx) => ({
      prob: model.predictProb(xVec),
      actual: y_all[idx],
      rawX: rows[idx].X
    }));

    let stage1Count = 0, stage2Count = 0, stage3Count = 0;
    let sumPD = 0.0;
    allScored.forEach((item) => {
      sumPD += item.prob;
      if (item.prob < 0.08) stage1Count++;
      else if (item.prob < 0.25) stage2Count++;
      else stage3Count++;
    });

    const avgPD = sumPD / (allScored.length || 1);
    const totalEAD = allScored.length * eadK;
    const baselPortfolio = computeBaselIRBCapital(avgPD, lgdVal, totalEAD);

    // Risk Deciles (D1 = Lowest Risk AAA to D10 = Highest Risk CCC/D)
    const sortedAll = allScored.slice().sort((a, b) => a.prob - b.prob);
    const gradeLabels = ["AAA", "AA", "A+", "A", "BBB+", "BBB", "BB+", "BB", "B", "CCC/D"];
    const decileTable = [];
    for (let d = 0; d < 10; d++) {
      const sIdx = Math.floor((d * sortedAll.length) / 10);
      const eIdx = Math.floor(((d + 1) * sortedAll.length) / 10);
      const bucket = sortedAll.slice(sIdx, eIdx);
      if (!bucket.length) continue;
      const bCnt = bucket.length;
      const bDefs = bucket.filter((it) => it.actual === 1).length;
      const bAvgPD = bucket.reduce((a, b) => a + b.prob, 0) / bCnt;
      const bObsBad = bDefs / bCnt;
      const bBasel = computeBaselIRBCapital(bAvgPD, lgdVal, bCnt * eadK);
      decileTable.push({
        decile: `D${d + 1} (${gradeLabels[d]})`,
        rating: gradeLabels[d],
        count: bCnt,
        defaults: bDefs,
        predicted_pd_pct: Number((bAvgPD * 100).toFixed(2)),
        observed_bad_pct: Number((bObsBad * 100).toFixed(2)),
        ecl_k: Number(bBasel.ecl.toFixed(2)),
        rwa_k: Number(bBasel.rwa.toFixed(2))
      });
    }

    // Banking Underwriting P&L Impact ($k)
    const interestGainPerGood = eadK * (baseRatePct / 100.0);
    const lossPerDefault = eadK * lgdVal;
    const netInterestEarnedK = Number((tn * interestGainPerGood).toFixed(2));
    const creditLossIncurredK = Number((fn * lossPerDefault).toFixed(2));
    const lossPreventedK = Number((tp * lossPerDefault).toFixed(2));
    const oppCostRejectedK = Number((fp * interestGainPerGood).toFixed(2));
    const netPortfolioPnlK = Number((netInterestEarnedK - creditLossIncurredK).toFixed(2));

    // 2D Decision Surface Grid
    const axX = Math.max(0, Math.min(axisX, nFeats - 1));
    const axY = Math.max(0, Math.min(axisY, nFeats - 1));
    const padX = (maxs[axX] - mins[axX]) * 0.10 || 1.0;
    const padY = (maxs[axY] - mins[axY]) * 0.10 || 1.0;
    const x0 = mins[axX] - padX, x1 = maxs[axX] + padX;
    const y0 = mins[axY] - padY, y1 = maxs[axY] + padY;
    const resolution = 56;
    const probs = [];

    for (let r = 0; r < resolution; r++) {
      const rowProbs = [];
      const wy = y0 + (r / (resolution - 1)) * (y1 - y0);
      for (let c = 0; c < resolution; c++) {
        const wx = x0 + (c / (resolution - 1)) * (x1 - x0);
        const vec = [...means];
        vec[axX] = wx;
        vec[axY] = wy;
        rowProbs.push(Number(model.predictProb(scaleVec(vec)).toFixed(3)));
      }
      probs.push(rowProbs);
    }

    const points = rows.map((r) => ({
      x: Number(r.X[axX].toFixed(4)),
      y: Number(r.X[axY].toFixed(4)),
      label: r.y
    }));

    const c0Count = rows.filter((r) => r.y === 0).length;
    const c1Count = rows.length - c0Count;

    const previewRows = rows.slice(0, 10).map((r) => {
      const obj = {};
      featureNames.forEach((fn, idx) => { obj[fn] = Number(r.X[idx].toFixed(2)); });
      const pVal = model.predictProb(scaleVec(r.X));
      const sg = pdToCreditScoreAndGrade(pVal);
      obj["Pred_PD_Pct"] = `${(pVal * 100).toFixed(1)}%`;
      obj["Credit_Score"] = sg.score;
      obj["Rating"] = sg.grade;
      obj[targetName] = r.y;
      return obj;
    });

    const featureRanges = featureNames.map((name, idx) => ({
      index: idx,
      name,
      min: Number(mins[idx].toFixed(2)),
      max: Number(maxs[idx].toFixed(2)),
      mean: Number(means[idx].toFixed(2)),
      iv: ivScores[idx]
    }));

    return {
      predictSingle: (rawFeatureValues, customLoanK, customLgdPct, customBaseRate) => {
        const p1 = model.predictProb(scaleVec(rawFeatureValues));
        const p0 = 1.0 - p1;
        const predClass = p1 >= threshold ? 1 : 0;
        const useEad = customLoanK !== undefined ? customLoanK : eadK;
        const useLgd = (customLgdPct !== undefined ? customLgdPct : lgdPct) / 100.0;
        const useBase = customBaseRate !== undefined ? customBaseRate : baseRatePct;

        const sg = pdToCreditScoreAndGrade(p1);
        const basel = computeBaselIRBCapital(p1, useLgd, useEad);

        const expLossRatePct = p1 * useLgd * 100.0;
        const capCostPct = (basel.regulatory_capital_8pct / Math.max(0.01, useEad)) * 12.0;
        const recAprPct = useBase + expLossRatePct + capCostPct + 1.45;
        const netIncomeK = useEad * ((recAprPct - useBase) / 100.0) - basel.ecl;
        const econCapK = Math.max(0.05 * useEad, basel.regulatory_capital_8pct);
        const rarocPct = (netIncomeK / econCapK) * 100.0;

        let decision = "DECLINE — EXCEEDS PD CUTOFF";
        if (p1 < threshold * 0.65) decision = "APPROVE — STAGE 1 PRIME";
        else if (p1 < threshold) decision = "CONDITIONAL APPROVE — PRICING COVENANT";

        return {
          predicted_class: predClass,
          predicted_label: classNames[predClass],
          decision,
          prob_performing: Number(p0.toFixed(4)),
          prob_default: Number(p1.toFixed(4)),
          credit_score: sg.score,
          rating_grade: sg.grade,
          ifrs9_stage: sg.stage,
          threshold,
          basel,
          recommended_apr_pct: Number(recAprPct.toFixed(2)),
          risk_spread_bps: Math.round((recAprPct - useBase) * 100),
          raroc_pct: Number(rarocPct.toFixed(1))
        };
      },
      responsePayload: {
        model_type: modelType,
        dataset_summary: {
          total_rows: n,
          train_rows: trainRows.length,
          test_rows: testRows.length,
          feature_names: featureNames,
          target_name: targetName,
          class_names: classNames,
          class_counts: { class_0: c0Count, class_1: c1Count },
          preview_rows: previewRows
        },
        metrics: {
          train_accuracy: Number(trainAcc.toFixed(4)),
          test_accuracy: Number(testAcc.toFixed(4)),
          precision: Number(precision.toFixed(4)),
          recall: Number(recall.toFixed(4)),
          f1_score: Number(f1.toFixed(4)),
          roc_auc: Number(auc.toFixed(4)),
          gini_coefficient: Number(giniCoeff.toFixed(4)),
          ks_statistic: Number(maxKS.toFixed(4)),
          ks_optimal_threshold: Number(ksOptTh.toFixed(2)),
          brier_score: Number(brierScore.toFixed(4)),
          confusion_matrix: { tn, fp, fn, tp }
        },
        basel_summary: {
          ...baselPortfolio,
          stage_1_count: stage1Count,
          stage_2_count: stage2Count,
          stage_3_count: stage3Count,
          net_interest_earned_k: netInterestEarnedK,
          credit_loss_incurred_k: creditLossIncurredK,
          loss_prevented_k: lossPreventedK,
          opp_cost_rejected_k: oppCostRejectedK,
          net_portfolio_pnl_k: netPortfolioPnlK
        },
        loss_history: model.lossHistory,
        roc_curve: rocCurve,
        ks_curve: ksCurve,
        decile_table: decileTable,
        weights: model.getFeatureWeights(),
        iv_scores: ivScores,
        feature_ranges: featureRanges,
        boundary_grid: {
          axis_x: axX,
          axis_y: axY,
          axis_x_name: featureNames[axX],
          axis_y_name: featureNames[axY],
          x_min: x0,
          x_max: x1,
          y_min: y0,
          y_max: y1,
          resolution,
          threshold,
          probs,
          points
        }
      }
    };
  }

  return {
    trainPipeline,
    computeBaselIRBCapital,
    pdToCreditScoreAndGrade
  };
})();

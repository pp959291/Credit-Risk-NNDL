/**
 * static/js/data_generator.js
 * Specialized Banking & Credit Risk Loan Portfolio Generator and CSV Parser.
 */
window.DataGenerator = (function () {
  const PORTFOLIO_META = {
    retail_unsecured: {
      title: "Retail Unsecured Personal Loans",
      defaultLGD: 65,
      avgEAD: 18.5,
      baseRate: 6.5,
      featureNames: [
        "Debt_to_Income_Pct",
        "Credit_Utilization_Pct",
        "Delinquencies_2Y",
        "Credit_History_Yrs",
        "Loan_to_Income_Ratio",
        "Bureau_Score_Proxy"
      ],
      targetName: "Default_12M",
      classNames: ["Performing (0)", "Defaulted (1)"]
    },
    mortgage_hl: {
      title: "Residential Mortgage & Home Equity (Basel III)",
      defaultLGD: 25,
      avgEAD: 245.0,
      baseRate: 4.85,
      featureNames: [
        "LTV_Ratio_Pct",
        "Debt_Service_DSCR",
        "Borrower_FICO_Score",
        "Rate_Shock_Bps",
        "Employment_Tenure_Yrs",
        "Collateral_Haircut_Pct"
      ],
      targetName: "Default_12M",
      classNames: ["Performing (0)", "Defaulted (1)"]
    },
    sme_commercial: {
      title: "SME & Corporate Commercial Credit Book",
      defaultLGD: 45,
      avgEAD: 420.0,
      baseRate: 5.75,
      featureNames: [
        "Net_Debt_to_EBITDA",
        "Interest_Coverage_ICR",
        "Current_Liquidity_Ratio",
        "EBITDA_Margin_Pct",
        "Payable_Delays_Days",
        "Asset_Turnover_Ratio"
      ],
      targetName: "Default_12M",
      classNames: ["Performing (0)", "Defaulted (1)"]
    },
    credit_cards: {
      title: "Revolving Credit Card & Overdraft Book",
      defaultLGD: 78,
      avgEAD: 8.2,
      baseRate: 11.2,
      featureNames: [
        "Revolving_Utilization_Pct",
        "Min_Payment_Freq_Pct",
        "Cash_Advance_Ratio_Pct",
        "Hard_Inquiries_6M",
        "Seasoning_Months",
        "Overlimit_Count_12M"
      ],
      targetName: "Default_12M",
      classNames: ["Performing (0)", "Defaulted (1)"]
    }
  };

  let rngSeed = 42;
  function seededRandom() {
    rngSeed = (rngSeed * 1664525 + 1013904223) % 4294967296;
    return (rngSeed >>> 0) / 4294967296;
  }

  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = seededRandom();
    while (v === 0) v = seededRandom();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  function clip(val, low, high) {
    return Math.max(low, Math.min(high, val));
  }

  function generateSynthetic(preset, nSamples, stressNoise, customSeed) {
    rngSeed = customSeed || 42;
    const key = PORTFOLIO_META[preset] ? preset : "retail_unsecured";
    const meta = PORTFOLIO_META[key];
    const rawItems = [];

    for (let i = 0; i < nSamples; i++) {
      let X = [];
      let z = 0;

      if (key === "retail_unsecured") {
        const dti = clip(34.0 + 13.5 * randn() + stressNoise * 12.0 * randn(), 6.0, 75.0);
        const util = clip(48.0 + 24.0 * randn() + stressNoise * 10.0 * randn(), 1.0, 99.5);
        const delinq = clip(Math.round(Math.abs(0.55 + (0.75 + stressNoise * 1.2) * randn() - 0.35)), 0, 7);
        const histYrs = clip(8.5 + 5.2 * randn(), 0.5, 30.0);
        const lti = clip(0.12 + 0.006 * dti + (seededRandom() * 0.30 - 0.08), 0.04, 0.95);
        const bureau = clip(760.0 - 1.45 * util - 1.25 * dti - 26.0 * delinq + 2.8 * histYrs + 18.0 * randn(), 480.0, 850.0);

        z = -3.35
          + 0.052 * (dti - 30.0)
          + 0.038 * (util - 45.0)
          + 0.00075 * Math.max(0, dti - 32.0) * Math.max(0, util - 50.0)
          + 0.68 * delinq
          - 0.11 * histYrs
          + 2.45 * (lti - 0.30)
          - 0.012 * (bureau - 660.0)
          + stressNoise * 2.1 * randn();
        X = [dti, util, delinq, histYrs, lti, bureau];

      } else if (key === "mortgage_hl") {
        const ltv = clip(74.0 + 14.0 * randn() + stressNoise * 10.0 * randn(), 35.0, 108.0);
        const dscr = clip(1.52 + 0.42 * randn() - stressNoise * 0.25 * randn(), 0.65, 3.10);
        const fico = clip(785.0 - 1.3 * Math.max(0, ltv - 60.0) + 38.0 * (dscr - 1.2) + 24.0 * randn(), 500.0, 850.0);
        const rateShock = clip(40.0 + seededRandom() * 300.0 + stressNoise * 75.0 * randn(), 0.0, 500.0);
        const tenure = clip(7.0 + 4.5 * randn(), 0.5, 28.0);
        const haircut = clip(18.0 + 0.22 * Math.max(0, ltv - 65.0) + 4.5 * randn(), 10.0, 58.0);

        z = -3.10
          + 0.068 * (ltv - 75.0)
          - 2.35 * (dscr - 1.35)
          + 0.042 * Math.max(0, ltv - 80.0) * Math.max(0, 1.30 - dscr)
          - 0.014 * (fico - 680.0)
          + 0.0065 * (rateShock - 150.0)
          - 0.095 * tenure
          + 0.035 * (haircut - 25.0)
          + stressNoise * 2.0 * randn();
        X = [ltv, dscr, fico, rateShock, tenure, haircut];

      } else if (key === "sme_commercial") {
        const debtEbitda = clip(3.2 + 1.65 * randn() + stressNoise * 0.8 * randn(), 0.2, 8.5);
        const icr = clip(6.2 - 0.72 * debtEbitda + 1.1 * randn(), 0.4, 11.0);
        const currRatio = clip(1.55 + 0.52 * randn(), 0.45, 3.60);
        const ebitdaMargin = clip(14.5 + 7.5 * randn() - stressNoise * 4.0 * randn(), -8.0, 42.0);
        const payDelays = clip(Math.round(Math.abs(12.0 + 16.0 * randn()) + 4.5 * debtEbitda), 0.0, 120.0);
        const turnover = clip(2.4 + 0.85 * randn(), 0.5, 6.0);

        z = -2.85
          + 0.68 * (debtEbitda - 3.2)
          - 0.58 * (icr - 2.5)
          + 0.35 * Math.max(0, debtEbitda - 3.5) * Math.max(0, 2.2 - icr)
          - 0.85 * (currRatio - 1.35)
          - 0.065 * (ebitdaMargin - 12.0)
          + 0.028 * (payDelays - 22.0)
          - 0.25 * (turnover - 2.0)
          + stressNoise * 2.1 * randn();
        X = [debtEbitda, icr, currRatio, ebitdaMargin, payDelays, turnover];

      } else {
        const revUtil = clip(52.0 + 26.0 * randn() + stressNoise * 12.0 * randn(), 1.0, 108.0);
        const minPay = clip(0.22 * revUtil + seededRandom() * 60.0, 0.0, 100.0);
        const cashAdv = clip(Math.abs(10.0 + 11.0 * randn()) + 0.14 * revUtil, 0.0, 75.0);
        const inquiries = clip(Math.round(Math.abs(1.4 + (1.5 + stressNoise * 2.0) * randn())), 0, 10);
        const seasoning = clip(46.0 + 26.0 * randn(), 3.0, 150.0);
        const overlimit = clip(Math.round((revUtil > 82 ? 1.4 : 0.2) + Math.abs(0.6 * randn())), 0, 6);

        z = -3.20
          + 0.042 * (revUtil - 50.0)
          + 0.028 * (minPay - 40.0)
          + 0.036 * (cashAdv - 15.0)
          + 0.42 * (inquiries - 1.5)
          - 0.018 * (seasoning - 36.0)
          + 0.72 * overlimit
          + stressNoise * 2.0 * randn();
        X = [revUtil, minPay, cashAdv, inquiries, seasoning, overlimit];
      }

      const p = 1.0 / (1.0 + Math.exp(-clip(z, -25, 25)));
      rawItems.push({ X: X.map((v) => Number(v.toFixed(3))), p });
    }

    const sortedProbs = rawItems.map((it) => it.p).slice().sort((a, b) => a - b);
    const q72 = sortedProbs[Math.floor(sortedProbs.length * 0.72)] || 0.45;
    const cutoff = Math.min(0.48, q72);

    const rows = rawItems.map((it) => ({
      X: it.X,
      y: it.p >= cutoff ? 1 : 0
    }));

    return {
      preset: key,
      meta,
      rows,
      featureNames: meta.featureNames,
      targetName: meta.targetName,
      classNames: meta.classNames
    };
  }

  function getSampleDataset(name) {
    const key = PORTFOLIO_META[name] ? name : "retail_unsecured";
    return generateSynthetic(key, 180, 0.14, 101);
  }

  function parseRawCSV(csvText) {
    const lines = csvText.trim().split(/\r?\n/).filter((l) => l.trim().length > 0);
    if (lines.length < 2) throw new Error("CSV must have a header and at least one data row.");
    const delim = (lines[0].includes(";") && !lines[0].includes(",")) ? ";" : ",";
    const headers = lines[0].split(delim).map((h) => h.trim().replace(/^"|"$/g, ""));
    const rawRows = lines.slice(1).map((line) => line.split(delim).map((c) => c.trim().replace(/^"|"$/g, "")));
    return { headers, rawRows };
  }

  function parseCSVToDataset(csvText, targetColName = null) {
    const { headers, rawRows } = parseRawCSV(csvText);
    let targetIdx = headers.length - 1;
    if (targetColName && headers.indexOf(targetColName) !== -1) {
      targetIdx = headers.indexOf(targetColName);
    }
    const featIndices = headers.map((_, i) => i).filter((i) => i !== targetIdx);
    const featureNames = featIndices.map((i) => headers[i]);
    const targetName = headers[targetIdx];

    const rawTargets = rawRows.map((r) => r[targetIdx]);
    const uniqueTargets = Array.from(new Set(rawTargets)).sort();
    const posVal = uniqueTargets.length >= 2 ? uniqueTargets[1] : "1";
    const classNames = uniqueTargets.length >= 2 && !(uniqueTargets[0] === "0" && uniqueTargets[1] === "1")
      ? [`${uniqueTargets[0]} (0)`, `${uniqueTargets[1]} (1)`]
      : ["Performing (0)", "Defaulted (1)"];

    const rows = rawRows.map((r) => {
      const X = featIndices.map((fi) => {
        const v = parseFloat(r[fi]);
        return isNaN(v) ? 0.0 : v;
      });
      const y = (r[targetIdx] === posVal || r[targetIdx] === "1") ? 1 : 0;
      return { X, y };
    });

    return {
      preset: "custom_csv",
      meta: {
        title: "Custom Uploaded Credit Portfolio",
        defaultLGD: 45,
        avgEAD: 25.0,
        baseRate: 6.0
      },
      rows,
      featureNames,
      targetName,
      classNames,
      headers
    };
  }

  return {
    PORTFOLIO_META,
    generateSynthetic,
    getSampleDataset,
    parseRawCSV,
    parseCSVToDataset
  };
})();

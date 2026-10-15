/**
 * RECS Energy Predictor & Simulation Studio
 * Frontend Application Controller - Lucide Icons & Indian Rupees (INR ₹)
 */

// Global State
let currentBaseline = null;
let whatifTimeout = null;

// Helper for Indian Rupee Formatting
function formatINR(val) {
    if (val === null || val === undefined || isNaN(val)) return '₹0';
    return '₹' + Number(Math.round(val)).toLocaleString('en-IN');
}

function formatINRCents(val) {
    if (val === null || val === undefined || isNaN(val)) return '₹0.00';
    return '₹' + Number(val).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initPredictor();
    initWhatIfSimulator();
    loadEvaluationData();
    loadHarmonizationData();
    refreshIcons();
});

function refreshIcons() {
    if (window.lucide && typeof window.lucide.createIcons === 'function') {
        window.lucide.createIcons();
    }
}

// ── 1. Tab Navigation ─────────────────────────────────────────
function initTabs() {
    const buttons = document.querySelectorAll('.tab-btn');
    const contents = document.querySelectorAll('.tab-content');

    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabTarget = btn.getAttribute('data-tab');

            buttons.forEach(b => b.classList.remove('active'));
            contents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            const targetEl = document.getElementById(tabTarget);
            if (targetEl) targetEl.classList.add('active');
            refreshIcons();
        });
    });
}

// ── 2. Energy Predictor (Questionnaire) ────────────────────────
function initPredictor() {
    const calcBtn = document.getElementById('calcBtn');

    calcBtn.addEventListener('click', (e) => {
        e.preventDefault();
        runPrediction();
    });

    // Run initial prediction with default values
    runPrediction();
}

function getFormPayload() {
    return {
        rooms: parseFloat(document.getElementById('inp_rooms').value) || 6,
        bedrooms: parseFloat(document.getElementById('inp_bedrooms').value) || 3,
        sqft: parseFloat(document.getElementById('inp_sqft').value) || 1800,
        housing_type: parseInt(document.getElementById('inp_housing_type').value) || 1,
        occupants: parseFloat(document.getElementById('inp_occupants').value) || 2,
        
        num_fridge: parseFloat(document.getElementById('inp_fridge').value) || 1,
        has_washer: document.getElementById('inp_washer').checked ? 1 : 0,
        has_dryer: document.getElementById('inp_dryer').checked ? 1 : 0,
        has_freezer: document.getElementById('inp_freezer').checked ? 1 : 0,
        has_ac: document.getElementById('inp_ac').checked ? 1 : 0,

        num_lights: parseFloat(document.getElementById('inp_lights').value) || 20,
        num_led: parseFloat(document.getElementById('inp_led').value) || 10,
        num_cfl: parseFloat(document.getElementById('inp_cfl').value) || 4,

        heat_fuel: parseInt(document.getElementById('inp_heat_fuel').value) || 1,
        heat_equip: parseInt(document.getElementById('inp_heat_equip').value) || 6,
        ac_type: parseInt(document.getElementById('inp_ac_type').value) || 5,

        region: parseInt(document.getElementById('inp_region').value) || 2,
        climate_region: 3,
        division: 4,
        hdd65: parseFloat(document.getElementById('inp_hdd').value) || 4200,
        cdd65: parseFloat(document.getElementById('inp_cdd').value) || 1400,
        survey_year: 2020
    };
}

async function runPrediction() {
    const calcBtn = document.getElementById('calcBtn');
    calcBtn.innerHTML = '<i data-lucide="loader-2" class="spin"></i> <span>Computing Forecast...</span>';
    refreshIcons();
    calcBtn.disabled = true;

    try {
        const payload = getFormPayload();
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error(`API returned status ${res.status}`);
        const data = await res.json();
        
        currentBaseline = data;
        renderResults(data);

        // Sync initial values into the What-If simulator
        syncWithSimulator(payload);
    } catch (err) {
        console.error('Prediction failed:', err);
        alert('Could not complete prediction. Please verify the backend server is running.');
    } finally {
        calcBtn.innerHTML = '<i data-lucide="sparkles"></i> <span>Calculate Energy Forecast & Bill (INR)</span>';
        calcBtn.disabled = false;
        refreshIcons();
    }
}

function renderResults(data) {
    // 1. Metric Hero Cards
    document.getElementById('res_kwh').textContent = Number(data.kwh_predicted).toLocaleString('en-IN') + ' kWh';
    
    // Support both annual_cost_inr and fallback
    const annualCost = data.annual_cost_inr || data.annual_cost_usd;
    const monthlyCost = data.monthly_cost_inr || data.monthly_cost_usd;

    document.getElementById('res_annual_cost').textContent = formatINR(annualCost);
    document.getElementById('res_monthly_cost').textContent = formatINR(monthlyCost);

    const diffBadge = document.getElementById('res_diff_badge');
    const pDiff = data.percent_vs_national;
    if (pDiff > 0) {
        diffBadge.textContent = `+${pDiff}% vs Cohort Avg`;
        diffBadge.className = 'metric-sub negative';
    } else {
        diffBadge.textContent = `${pDiff}% vs Cohort Avg`;
        diffBadge.className = 'metric-sub';
    }

    // 2. Comparison Bar Visualizer
    const b = data.benchmarks;
    const maxVal = Math.max(data.kwh_predicted, b.national_mean_kwh, b.regional_mean_kwh, b.efficient_benchmark_kwh) * 1.15;

    setBarWidth('bar_user', 'val_user', data.kwh_predicted, maxVal, `${Number(data.kwh_predicted).toLocaleString('en-IN')} kWh (${formatINR(annualCost)})`);
    setBarWidth('bar_nat', 'val_nat', b.national_mean_kwh, maxVal, `${Number(b.national_mean_kwh).toLocaleString('en-IN')} kWh (${formatINR(b.national_mean_cost)})`);
    setBarWidth('bar_reg', 'val_reg', b.regional_mean_kwh, maxVal, `${Number(b.regional_mean_kwh).toLocaleString('en-IN')} kWh`);
    setBarWidth('bar_eff', 'val_eff', b.efficient_benchmark_kwh, maxVal, `${Number(b.efficient_benchmark_kwh).toLocaleString('en-IN')} kWh (${formatINR(b.efficient_benchmark_cost)})`);

    // 3. SHAP Feature Drivers
    const driversList = document.getElementById('shap_drivers_list');
    driversList.innerHTML = '';
    
    if (data.top_drivers && data.top_drivers.length > 0) {
        data.top_drivers.forEach(d => {
            const item = document.createElement('div');
            item.className = 'shap-item';
            item.innerHTML = `
                <div class="shap-item-left">
                    <span class="shap-rank">#${d.rank}</span>
                    <span class="shap-name">${d.label}</span>
                </div>
                <div class="shap-impact">±${Number(d.impact_kwh).toLocaleString('en-IN')} kWh/yr impact</div>
            `;
            driversList.appendChild(item);
        });
    }

    refreshIcons();
}

function setBarWidth(barId, valId, value, maxVal, text) {
    const barEl = document.getElementById(barId);
    const valEl = document.getElementById(valId);
    if (!barEl || !valEl) return;

    const pct = Math.min(Math.max((value / maxVal) * 100, 4), 100);
    barEl.style.width = pct + '%';
    valEl.textContent = text;
}

// ── 3. What-If Interactive Simulator ──────────────────────────
function initWhatIfSimulator() {
    const sliders = ['sim_fridge', 'sim_led_pct', 'sim_occupants', 'sim_thermostat'];
    sliders.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('input', () => {
                updateSliderBadge(id);
                triggerWhatIfDebounced();
            });
        }
    });

    const toggles = ['sim_freezer', 'sim_dryer', 'sim_ac', 'sim_washer', 'sim_heat_electric'];
    toggles.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('change', () => triggerWhatIfDebounced());
        }
    });
}

function updateSliderBadge(id) {
    const el = document.getElementById(id);
    const badge = document.getElementById(id + '_val');
    if (!el || !badge) return;

    if (id === 'sim_led_pct') {
        badge.textContent = el.value + '%';
    } else if (id === 'sim_thermostat') {
        const val = parseInt(el.value);
        if (val > 0) {
            badge.textContent = `+${val}% (High AC)`;
        } else if (val < 0) {
            badge.textContent = `${val}% (Eco Mode)`;
        } else {
            badge.textContent = `0% (Standard)`;
        }
    } else {
        badge.textContent = el.value;
    }
}

function syncWithSimulator(payload) {
    const fridgeSlider = document.getElementById('sim_fridge');
    const occSlider = document.getElementById('sim_occupants');
    const ledSlider = document.getElementById('sim_led_pct');
    const thermSlider = document.getElementById('sim_thermostat');

    if (fridgeSlider) {
        fridgeSlider.value = payload.num_fridge;
        updateSliderBadge('sim_fridge');
    }
    if (occSlider) {
        occSlider.value = payload.occupants;
        updateSliderBadge('sim_occupants');
    }
    if (ledSlider && payload.num_lights > 0) {
        const ledPct = Math.round((payload.num_led / payload.num_lights) * 100);
        ledSlider.value = ledPct;
        updateSliderBadge('sim_led_pct');
    }
    if (thermSlider) {
        thermSlider.value = 0;
        updateSliderBadge('sim_thermostat');
    }

    const freezTog = document.getElementById('sim_freezer');
    if (freezTog) freezTog.checked = payload.has_freezer === 1;

    const dryTog = document.getElementById('sim_dryer');
    if (dryTog) dryTog.checked = payload.has_dryer === 1;

    const acTog = document.getElementById('sim_ac');
    if (acTog) acTog.checked = payload.has_ac === 1;

    const washTog = document.getElementById('sim_washer');
    if (washTog) washTog.checked = payload.has_washer === 1;

    const heatTog = document.getElementById('sim_heat_electric');
    if (heatTog) heatTog.checked = payload.heat_fuel === 1;

    triggerWhatIf();
}

function triggerWhatIfDebounced() {
    clearTimeout(whatifTimeout);
    whatifTimeout = setTimeout(triggerWhatIf, 120);
}

async function triggerWhatIf() {
    if (!currentBaseline) return;

    const basePayload = getFormPayload();

    // Override with simulator slider settings
    const simFridge = parseFloat(document.getElementById('sim_fridge').value) || 1;
    const simOcc = parseFloat(document.getElementById('sim_occupants').value) || 2;
    const simLedPct = parseFloat(document.getElementById('sim_led_pct').value) || 50;
    const simThermostatPct = parseFloat(document.getElementById('sim_thermostat')?.value || 0);

    const simFreezer = document.getElementById('sim_freezer')?.checked ? 1 : 0;
    const simDryer = document.getElementById('sim_dryer')?.checked ? 1 : 0;
    const simAc = document.getElementById('sim_ac')?.checked ? 1 : 0;
    const simWasher = document.getElementById('sim_washer')?.checked ? 1 : 0;
    const simHeatElectric = document.getElementById('sim_heat_electric')?.checked ? 1 : 0;

    const totalLights = basePayload.num_lights || 20;
    const simLed = Math.round((simLedPct / 100) * totalLights);
    const simCfl = Math.max(0, totalLights - simLed);

    // Adjust CDD (cooling degree days) based on thermostat & AC switch
    let adjustedCdd = basePayload.cdd65;
    if (simAc === 0) {
        adjustedCdd = 0;
    } else {
        adjustedCdd = Math.max(0, basePayload.cdd65 * (1 + simThermostatPct / 100));
    }

    const payload = {
        ...basePayload,
        num_fridge: simFridge,
        occupants: simOcc,
        num_led: simLed,
        num_cfl: simCfl,
        has_freezer: simFreezer,
        has_dryer: simDryer,
        has_washer: simWasher,
        has_ac: simAc,
        cdd65: adjustedCdd,
        heat_fuel: simHeatElectric
    };

    try {
        const res = await fetch('/api/whatif', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!res.ok) return;
        const simData = await res.json();

        renderWhatIfResults(simData);
    } catch (e) {
        console.error('What-if request failed:', e);
    }
}

function renderWhatIfResults(simData) {
    const baseKwh = currentBaseline.kwh_predicted;
    const baseCost = currentBaseline.annual_cost_inr || currentBaseline.annual_cost_usd;

    const simKwh = simData.kwh_predicted;
    const simCost = simData.annual_cost_inr || simData.annual_cost_usd;

    const deltaKwh = Math.round(simKwh - baseKwh);
    const deltaCost = Math.round(simCost - baseCost);

    document.getElementById('sim_kwh_val').textContent = Number(simKwh).toLocaleString('en-IN') + ' kWh';
    document.getElementById('sim_cost_val').textContent = formatINR(simCost) + ' / yr';

    const deltaBox = document.getElementById('sim_delta_badge');
    if (deltaCost < 0) {
        deltaBox.className = 'delta-badge saving';
        deltaBox.innerHTML = `🎉 Estimated Savings: ${formatINR(Math.abs(deltaCost))}/yr (${Number(Math.abs(deltaKwh)).toLocaleString('en-IN')} kWh saved)`;
    } else if (deltaCost > 0) {
        deltaBox.className = 'delta-badge costing';
        deltaBox.innerHTML = `⚠️ Added Cost: +${formatINR(deltaCost)}/yr (+${Number(deltaKwh).toLocaleString('en-IN')} kWh)`;
    } else {
        deltaBox.className = 'delta-badge';
        deltaBox.innerHTML = `No change from current baseline`;
    }
}

// ── 4. Evaluation & Benchmarks Tab ────────────────────────────
async function loadEvaluationData() {
    try {
        const res = await fetch('/api/metrics');
        if (!res.ok) return;
        const data = await res.json();

        const tbody = document.getElementById('metrics_table_body');
        if (!tbody || !data.models) return;

        tbody.innerHTML = '';
        data.models.forEach(m => {
            const tr = document.createElement('tr');
            const isChampion = m.model.includes('XGBoost');
            tr.innerHTML = `
                <td><strong>${m.model}</strong> ${isChampion ? '<span class="badge-champion">Best Model</span>' : ''}</td>
                <td>${Math.round(m.MAE).toLocaleString('en-IN')} kWh</td>
                <td>${Math.round(m.RMSE).toLocaleString('en-IN')} kWh</td>
                <td><strong>${Number(m.R2).toFixed(3)}</strong></td>
                <td>${Number(m.MAPE).toFixed(1)}%</td>
            `;
            tbody.appendChild(tr);
        });
        refreshIcons();
    } catch (e) {
        console.error('Failed loading metrics:', e);
    }
}

// ── 5. Harmonization & Methodology Tab ────────────────────────
async function loadHarmonizationData() {
    try {
        const res = await fetch('/api/harmonization');
        if (!res.ok) return;
        const data = await res.json();

        const tbody = document.getElementById('harmonization_table_body');
        const searchInput = document.getElementById('mapping_search');

        function renderRows(items) {
            tbody.innerHTML = '';
            items.forEach(item => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>${item.feature}</strong></td>
                    <td><code>${item.clean_col}</code></td>
                    <td><code>${item['2005']}</code></td>
                    <td><code>${item['2009']}</code></td>
                    <td><code>${item['2015']}</code></td>
                    <td><code>${item['2020']}</code></td>
                    <td><small>${item.description}</small></td>
                `;
                tbody.appendChild(tr);
            });
            refreshIcons();
        }

        renderRows(data.variables);

        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                const q = e.target.value.toLowerCase();
                const filtered = data.variables.filter(v => 
                    v.feature.toLowerCase().includes(q) ||
                    v.clean_col.toLowerCase().includes(q) ||
                    v.description.toLowerCase().includes(q)
                );
                renderRows(filtered);
            });
        }
    } catch (e) {
        console.error('Failed loading harmonization:', e);
    }
}

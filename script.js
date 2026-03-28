/* =========================================================
   ML Credit Scorer — script.js
   All core logic preserved. Micro-interactions added on top.
   ========================================================= */

/* ── 1. Init on DOM ready ─────────────────────────────────── */
document.addEventListener('DOMContentLoaded', function () {

    /* Letter-stagger on h1 */
    var titleEl = document.getElementById('hero-title');
    if (titleEl) {
        var raw = titleEl.textContent;
        var html = '';
        for (var i = 0; i < raw.length; i++) {
            var ch = raw[i] === ' ' ? '&nbsp;' : raw[i];
            var delay = (i * 0.038).toFixed(3);
            html += '<span class="ch" style="animation-delay:' + delay + 's">' + ch + '</span>';
        }
        titleEl.innerHTML = html;
    }

    /* Accordion toggle */
    document.querySelectorAll('.accordion').forEach(function (acc) {
        var btn = acc.querySelector('.acc-header');
        btn.addEventListener('click', function () {
            acc.classList.toggle('active');
        });
    });

    /* Input flash micro-interaction */
    document.querySelectorAll('.field input, .field select').forEach(function (el) {
        el.addEventListener('change', function () {
            this.classList.remove('flashed');
            /* force reflow to restart animation */
            void this.offsetWidth;
            this.classList.add('flashed');
        });
    });
});

/* ── 2. Form Submit ────────────────────────────────────────── */
document.getElementById('predict-form').addEventListener('submit', async function (e) {
    e.preventDefault();

    var btn = document.getElementById('submit-btn');
    btn.classList.add('loading');
    btn.querySelector('.btn-label').textContent = 'PROCESSING...';
    btn.querySelector('.btn-icon').className = 'fa-solid fa-microchip fa-spin btn-icon';

    var payload = {
        Annual_Income:          parseFloat(document.getElementById('Annual_Income').value),
        Monthly_Inhand_Salary:  parseFloat(document.getElementById('Monthly_Inhand_Salary').value),
        Total_EMI_per_month:    parseFloat(document.getElementById('Total_EMI_per_month').value),
        Interest_Rate:          parseFloat(document.getElementById('Interest_Rate').value),
        Num_Credit_Card:        parseFloat(document.getElementById('Num_Credit_Card').value),
        Delay_from_due_date:    parseFloat(document.getElementById('Delay_from_due_date').value),
        Num_Credit_Inquiries:   parseFloat(document.getElementById('Num_Credit_Inquiries').value),
        Credit_Mix:             document.getElementById('Credit_Mix').value,
        Outstanding_Debt:       parseFloat(document.getElementById('Outstanding_Debt').value),
        Payment_of_Min_Amount:  document.getElementById('Payment_of_Min_Amount').value,
        Changed_Credit_Limit:   parseFloat(document.getElementById('Changed_Credit_Limit').value),
        Num_of_Loan:            parseFloat(document.getElementById('Num_of_Loan').value),
        Credit_History_Months:  parseFloat(document.getElementById('Credit_History_Months').value)
    };

    try {
        var res  = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        var data = await res.json();

        if (res.ok) {
            updateDashboard(data, payload);
        } else {
            alert('Backend error: ' + (data.detail || 'unknown'));
        }
    } catch (err) {
        console.error(err);
        alert('Could not reach ML backend.');
    } finally {
        setTimeout(function () {
            btn.classList.remove('loading');
            btn.querySelector('.btn-label').textContent = 'INITIALIZE PREDICTION';
            btn.querySelector('.btn-icon').className = 'fa-solid fa-microchip btn-icon';
        }, 600);
    }
});

/* ── 3. animateValue — counts up a numeric element ─────────── */
function animateValue(el, from, to, duration) {
    var start = null;
    function step(ts) {
        if (!start) start = ts;
        var prog = Math.min((ts - start) / duration, 1);
        el.textContent = Math.floor(prog * (to - from) + from);
        if (prog < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
}

/* ── 4. updateDashboard ────────────────────────────────────── */
function updateDashboard(data, inputs) {
    /* Score count-up */
    var scoreEl = document.getElementById('target-score');
    animateValue(scoreEl, 300, data.score, 1300);

    /* Radial ripple */
    var ring = document.getElementById('ripple-ring');
    ring.classList.remove('pop');
    void ring.offsetWidth;
    ring.classList.add('pop');

    /* 360 SVG gauge — circumference 2π×70 ≈ 439.8 */
    var fill    = document.getElementById('gauge-fill');
    var norm    = Math.max(0, Math.min((data.score - 300) / 600, 1));
    var offset  = 439.8 - norm * 439.8;
    setTimeout(function () { fill.style.strokeDashoffset = offset; }, 80);

    /* Band badge */
    var badge = document.getElementById('band-badge');
    var cls   = data.band.replace(/\s+/g, '.');
    badge.className = 'band-badge ' + cls;
    badge.textContent = data.band;

    /* Hex metric cards */
    var dti = data.metrics.dti;
    var emi = data.metrics.emi;
    var age = data.metrics.credit_age_years;

    document.getElementById('dti-val').textContent = dti.toFixed(1) + '%';
    document.getElementById('emi-val').textContent = emi.toFixed(1) + '%';
    document.getElementById('age-val').textContent = age.toFixed(1) + 'Y';

    document.getElementById('card-dti').className = 'hex-card' + (dti > 40 ? ' danger' : '');
    document.getElementById('card-emi').className = 'hex-card' + (emi > 40 ? ' danger' : '');
    document.getElementById('card-age').className = 'hex-card' + (age < 3  ? ' danger' : '');

    /* AI insights */
    generateExplanations(data, inputs);
}

/* ── 5. generateExplanations ───────────────────────────────── */
function generateExplanations(data, inputs) {
    var list = document.getElementById('rationale-list');
    list.innerHTML = '';

    var good = [];
    var help = [];

    var dti = (inputs.Outstanding_Debt / inputs.Annual_Income) * 100 || 0;
    var emi = (inputs.Total_EMI_per_month / inputs.Monthly_Inhand_Salary) * 100 || 0;

    /* DTI */
    if (dti < 30)
        good.push('Healthy Debt-to-Income (' + dti.toFixed(1) + '%) signals excellent borrowing capacity.');
    else
        help.push('Reduce outstanding debt — your DTI (' + dti.toFixed(1) + '%) exceeds the safe 30% threshold.');

    /* EMI burden */
    if (emi < 30)
        good.push('Low EMI burden (' + emi.toFixed(1) + '%) preserves monthly cash liquidity.');
    else
        help.push('EMI burden (' + emi.toFixed(1) + '%) is high. Consolidating loans could free up cash flow.');

    /* Payment delay */
    if (inputs.Delay_from_due_date <= 5)
        good.push('Excellent payment latency (avg ' + inputs.Delay_from_due_date + ' days) confirms strong financial discipline.');
    else
        help.push('Average ' + inputs.Delay_from_due_date + '-day payment delay detected. Enable auto-pay to resolve this.');

    /* Credit history */
    if (inputs.Credit_History_Months >= 60)
        good.push('Deep credit history (' + inputs.Credit_History_Months + ' months) provides a reliable statistical foundation.');
    else
        help.push('Credit history is short (' + inputs.Credit_History_Months + ' months). Aging accounts past 5 years boosts scores significantly.');

    /* Credit mix */
    if (inputs.Credit_Mix === 'Good')
        good.push('Diverse credit mix (Good) proves multi-dimensional debt management maturity.');
    else
        help.push('Monolithic credit type detected. Diversifying loan types (secured + unsecured) elevates the mix score.');

    /* Credit cards */
    if (inputs.Num_Credit_Card <= 3)
        good.push('Conservative card count (' + inputs.Num_Credit_Card + ') limits maximum revolving exposure.');
    else
        help.push('High card count (' + inputs.Num_Credit_Card + ') signals elevated credit exposure. Consider closing inactive accounts.');

    /* Inquiries */
    if (inputs.Num_Credit_Inquiries <= 2)
        good.push('Low inquiry count (' + inputs.Num_Credit_Inquiries + ') passes hard-pull desperation filters cleanly.');
    else
        help.push('Elevated inquiries (' + inputs.Num_Credit_Inquiries + '). Suspend new credit applications for 6 months to recover.');

    /* Min payment */
    if (inputs.Payment_of_Min_Amount === 'No')
        good.push('Full-balance payments eliminate revolving interest risk entirely.');
    else
        help.push('Paying only the minimum due signals cash-flow stress to lenders. Switch to full-balance clearing.');

    /* Interest rate */
    if (inputs.Interest_Rate < 12)
        good.push('Sub-12% interest rate reflects strong trust-level from existing lenders.');
    else
        help.push('High interest rate (' + inputs.Interest_Rate + '%). Refinancing current debt can mechanically lower your risk profile.');

    /* Score-based allocation */
    var tg, th;
    var s = data.score;
    if      (s >= 800) { tg = 5; th = 0; }
    else if (s >= 750) { tg = 4; th = 1; }
    else if (s >= 700) { tg = 3; th = 2; }
    else if (s >= 600) { tg = 2; th = 3; }
    else if (s >= 500) { tg = 1; th = 4; }
    else               { tg = 0; th = 5; }

    if (good.length < tg) { th += (tg - good.length); tg = good.length; }
    if (help.length < th) { tg += (th - help.length); th = help.length; }

    var items = [];
    for (var i = 0; i < tg; i++) items.push({ text: good[i], isGood: true });
    for (var j = 0; j < th; j++) items.push({ text: help[j], isGood: false });

    /* Render with stagger */
    items.forEach(function (item, idx) {
        var li    = document.createElement('li');
        li.className = 'tip-item ' + (item.isGood ? 'good' : 'help');
        li.style.animationDelay = (idx * 0.085 + 0.05) + 's';

        var icon = document.createElement('i');
        icon.className = 'fa-solid ' + (item.isGood ? 'fa-circle-check' : 'fa-triangle-exclamation');

        var span = document.createElement('span');
        span.textContent = item.text;

        li.appendChild(icon);
        li.appendChild(span);
        list.appendChild(li);
    });
}

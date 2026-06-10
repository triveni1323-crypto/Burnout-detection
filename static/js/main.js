// main.js — Interactive enhancements

// Animate probability bars on result page
window.addEventListener('load', () => {
    const bars = document.querySelectorAll('.bar-fill');
    bars.forEach(bar => {
        const width = bar.style.width;
        bar.style.width = '0%';
        setTimeout(() => { bar.style.width = width; }, 300);
    });
});

// Animate stat numbers on home page
function animateNumber(el, target, duration = 1500) {
    let start = 0;
    const step = target / (duration / 16);
    const timer = setInterval(() => {
        start += step;
        if (start >= target) { start = target; clearInterval(timer); }
        el.textContent = Math.floor(start) +
            (el.dataset.suffix || '');
    }, 16);
}

const statNums = document.querySelectorAll('.stat-number');
statNums.forEach(el => {
    const target = parseInt(el.dataset.target);
    if (target) animateNumber(el, target);
});

// Form input validation feedback
const inputs = document.querySelectorAll('.form-group input');
inputs.forEach(input => {
    input.addEventListener('input', () => {
        if (input.value && !isNaN(input.value)) {
            input.classList.add('input-valid');
            input.classList.remove('input-invalid');
        } else {
            input.classList.add('input-invalid');
            input.classList.remove('input-valid');
        }
    });
});

// Loading spinner on form submit
const form = document.querySelector('form');
if (form) {
    form.addEventListener('submit', () => {
        const btn = form.querySelector('button[type="submit"]');
        btn.innerHTML = '⏳ Analyzing... Please wait';
        btn.disabled = true;
    });
}
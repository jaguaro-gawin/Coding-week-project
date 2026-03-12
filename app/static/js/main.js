/* ═══════════════════════════════════════════════════════════
   PediAppend – main.js
   ═══════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {

    /* ── Navbar scroll ── */
    const nav = document.querySelector('.navbar');
    if (nav) {
        window.addEventListener('scroll', () => {
            nav.classList.toggle('scrolled', window.scrollY > 20);
        });
        nav.classList.toggle('scrolled', window.scrollY > 20);
    }

    /* ── Smooth scroll for anchors (non-landing pages only) ── */
    document.querySelectorAll('a[href^="#"]').forEach(a => {
        a.addEventListener('click', e => {
            if (a.hasAttribute('data-goto')) return; // handled by slide logic
            const t = document.querySelector(a.getAttribute('href'));
            if (t) { e.preventDefault(); t.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
        });
    });

    /* ── Intersection Observer (animate-in) ── */
    const io = new IntersectionObserver(entries => {
        entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('visible'); io.unobserve(e.target); } });
    }, { threshold: 0.08 });
    document.querySelectorAll('.animate-in').forEach(el => io.observe(el));

    /* ── Stat count-up animation ── */
    const statObserver = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (!entry.isIntersecting) return;
            entry.target.querySelectorAll('.stat-value[data-count]').forEach(el => {
                const target = parseFloat(el.dataset.count);
                const duration = 1500;
                const start = performance.now();
                function tick(now) {
                    const progress = Math.min((now - start) / duration, 1);
                    const eased = 1 - Math.pow(1 - progress, 3);
                    el.textContent = (target * eased).toFixed(1);
                    if (progress < 1) requestAnimationFrame(tick);
                }
                requestAnimationFrame(tick);
            });
            statObserver.unobserve(entry.target);
        });
    }, { threshold: 0.3 });
    document.querySelectorAll('.stats-grid').forEach(el => statObserver.observe(el));

    /* ── Landing page slide navigation ── */
    const sideSteps = document.querySelectorAll('.side-step');
    const landingSections = document.querySelectorAll('.landing-step');
    let currentSlide = 0;
    let slideAnimating = false;

    function goToSlide(index) {
        if (slideAnimating || index === currentSlide || index < 0 || index >= landingSections.length) return;
        slideAnimating = true;
        const goingDown = index > currentSlide;
        const prev = landingSections[currentSlide];
        const next = landingSections[index];

        // Exit current slide in opposite direction
        prev.classList.remove('active');
        prev.classList.add(goingDown ? 'slide-exit-up' : 'slide-exit-down');

        // Prepare entry direction
        next.style.transition = 'none';
        next.style.transform = goingDown ? 'translateY(40px)' : 'translateY(-40px)';
        next.style.opacity = '0';
        next.classList.add('active');

        // Force reflow then animate in
        void next.offsetHeight;
        next.style.transition = '';
        next.style.transform = '';
        next.style.opacity = '';

        // Update side dots
        sideSteps.forEach((s, i) => s.classList.toggle('active', i === index));

        // Trigger animate-in for new slide's elements
        next.querySelectorAll('.animate-in').forEach(el => el.classList.add('visible'));

        // Trigger stat count-up if entering stats step
        if (next.querySelector('.stats-grid')) {
            next.querySelectorAll('.stat-value[data-count]').forEach(el => {
                const target = parseFloat(el.dataset.count);
                const duration = 1500, start = performance.now();
                function tick(now) {
                    const progress = Math.min((now - start) / duration, 1);
                    const eased = 1 - Math.pow(1 - progress, 3);
                    el.textContent = (target * eased).toFixed(1);
                    if (progress < 1) requestAnimationFrame(tick);
                }
                requestAnimationFrame(tick);
            });
        }

        currentSlide = index;
        setTimeout(() => {
            prev.classList.remove('slide-exit-up', 'slide-exit-down');
            slideAnimating = false;
        }, 550);
    }

    if (landingSections.length) {
        // Side dot clicks
        sideSteps.forEach(s => {
            s.addEventListener('click', e => {
                e.preventDefault();
                goToSlide(parseInt(s.dataset.goto));
            });
        });

        // Arrow button clicks
        document.querySelectorAll('.slide-arrow-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                if (btn.dataset.dir === 'next') goToSlide(currentSlide + 1);
                else goToSlide(currentSlide - 1);
            });
        });

        // data-goto links (e.g. "How It Works" button in hero)
        document.querySelectorAll('[data-goto]').forEach(el => {
            if (el.classList.contains('side-step') || el.classList.contains('slide-arrow-btn')) return;
            el.addEventListener('click', e => {
                e.preventDefault();
                goToSlide(parseInt(el.dataset.goto));
            });
        });

        // Keyboard navigation
        document.addEventListener('keydown', e => {
            if (!document.querySelector('.slides-viewport')) return;
            if (e.key === 'ArrowDown' || e.key === 'ArrowRight') { e.preventDefault(); goToSlide(currentSlide + 1); }
            if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') { e.preventDefault(); goToSlide(currentSlide - 1); }
        });

        // Mouse wheel navigation (properly throttled)
        const viewport = document.getElementById('slidesViewport');
        let wheelCooldown = false;
        if (viewport) {
            viewport.addEventListener('wheel', e => {
                e.preventDefault();
                if (wheelCooldown || slideAnimating) return;
                wheelCooldown = true;
                if (e.deltaY > 0) goToSlide(currentSlide + 1);
                else goToSlide(currentSlide - 1);
                setTimeout(() => { wheelCooldown = false; }, 800);
            }, { passive: false });
        }

        // Init: make sure first slide is visible
        landingSections[0].classList.add('active');
        landingSections[0].querySelectorAll('.animate-in').forEach(el => el.classList.add('visible'));
    }

    /* ── Particles ── */
    const pc = document.getElementById('particles');
    if (pc) {
        for (let i = 0; i < 20; i++) {
            const d = document.createElement('span');
            d.className = 'particle';
            d.style.left = Math.random() * 100 + '%';
            d.style.top = Math.random() * 100 + '%';
            d.style.animationDelay = Math.random() * 6 + 's';
            d.style.animationDuration = (4 + Math.random() * 6) + 's';
            d.style.width = d.style.height = (2 + Math.random() * 4) + 'px';
            pc.appendChild(d);
        }
    }

    /* ═══ MULTI-STEP FORM ═══ */
    const steps = document.querySelectorAll('.form-step');
    const nextBtn = document.getElementById('nextBtn');
    const prevBtn = document.getElementById('prevBtn');
    let currentStep = 0;

    function showStep(n) {
        steps.forEach((s, i) => s.style.display = i === n ? '' : 'none');
        if (prevBtn) prevBtn.style.visibility = n === 0 ? 'hidden' : 'visible';
        if (nextBtn) {
            if (n === steps.length - 1) {
                nextBtn.innerHTML = '&#129657; Analyze &rarr;';
                nextBtn.classList.add('submit-mode');
            } else {
                nextBtn.innerHTML = 'Next &rarr;';
                nextBtn.classList.remove('submit-mode');
            }
        }
        /* Update step indicators */
        document.querySelectorAll('.step-circle').forEach((c, i) => {
            c.classList.remove('active', 'completed', 'pending');
            if (i < n) c.classList.add('completed');
            else if (i === n) c.classList.add('active');
            else c.classList.add('pending');
        });
        document.querySelectorAll('.step-label-text').forEach((l, i) => {
            l.style.color = i <= n ? 'var(--slate-800)' : 'var(--slate-400)';
        });
        /* Progress bar */
        const fill = document.querySelector('.step-progress-fill');
        if (fill) fill.style.width = ((n + 1) / steps.length * 100) + '%';
        /* Connector fills */
        document.querySelectorAll('.step-connector-fill').forEach((cf, i) => {
            cf.style.width = i < n ? '100%' : '0%';
        });
    }

    function validateStep(n) {
        const step = steps[n];
        if (!step) return true;
        const required = step.querySelectorAll('[required]');
        let ok = true;
        required.forEach(el => {
            const g = el.closest('.form-group') || el.closest('.bmi-card') || el.parentElement;
            if (!el.value) { g.classList.add('has-error'); ok = false; }
            else g.classList.remove('has-error');
        });
        /* Sex field */
        const sexInput = step.querySelector('#Sex');
        if (sexInput && !sexInput.value) {
            const g = sexInput.closest('.form-group') || sexInput.parentElement;
            g.classList.add('has-error');
            ok = false;
        }
        return ok;
    }

    if (nextBtn) {
        nextBtn.addEventListener('click', () => {
            if (!validateStep(currentStep)) return;
            if (currentStep === steps.length - 1) {
                document.getElementById('diagnosisForm').submit();
            } else {
                currentStep++;
                showStep(currentStep);
                window.scrollTo({ top: document.querySelector('.form-section').offsetTop - 100, behavior: 'smooth' });
            }
        });
    }
    if (prevBtn) {
        prevBtn.addEventListener('click', () => {
            if (currentStep > 0) { currentStep--; showStep(currentStep); }
        });
    }
    if (steps.length) showStep(0);

    /* ── Age calc from dates ── */
    const dobEl = document.getElementById('DateOfBirth');
    const examEl = document.getElementById('ExamDate');
    const ageEl = document.getElementById('Age');
    const ageDisp = document.getElementById('ageDisplay');
    const ageFill = document.getElementById('ageFill');
    const ageCard = document.getElementById('ageCard');

    function calcAge() {
        if (!dobEl?.value || !examEl?.value) return;
        const dob = new Date(dobEl.value);
        const exam = new Date(examEl.value);
        if (exam <= dob) return;
        const diffMs = exam - dob;
        const age = diffMs / (365.25 * 24 * 60 * 60 * 1000);
        const val = age.toFixed(1);
        if (ageDisp) ageDisp.textContent = val;
        if (ageFill) ageFill.style.width = Math.min(age / 18 * 100, 100) + '%';
        if (ageCard) ageCard.classList.add('active');
        const ageErrorEl = document.getElementById('ageError');
        if (age > 18) {
            if (ageDisp) ageDisp.style.color = 'var(--rose-500, #f43f5e)';
            if (ageErrorEl) ageErrorEl.style.display = 'block';
            if (ageEl) ageEl.value = '';
        } else {
            if (ageDisp) ageDisp.style.color = '';
            if (ageErrorEl) ageErrorEl.style.display = 'none';
            if (ageEl) ageEl.value = val;
            const g = ageEl?.closest('.form-group') || ageCard;
            if (g) g.classList.remove('has-error');
        }
    }
    if (dobEl) dobEl.addEventListener('input', calcAge);
    if (examEl) {
        examEl.addEventListener('input', calcAge);
        examEl.value = new Date().toISOString().split('T')[0];
    }

    /* ── BMI calc ── */
    const hEl = document.getElementById('Height');
    const wEl = document.getElementById('Weight');
    const bmiEl = document.getElementById('BMI');
    const bmiDisp = document.getElementById('bmiDisplay');
    const bmiFill = document.getElementById('bmiFill');
    const bmiCard = document.getElementById('bmiCard');

    function calcBMI() {
        const h = parseFloat(hEl?.value);
        const w = parseFloat(wEl?.value);
        if (h > 0 && w > 0) {
            const bmi = w / ((h / 100) ** 2);
            const val = bmi.toFixed(1);
            if (bmiEl) bmiEl.value = val;
            if (bmiDisp) bmiDisp.textContent = val;
            if (bmiFill) bmiFill.style.width = Math.min(bmi / 40 * 100, 100) + '%';
            if (bmiCard) bmiCard.classList.add('active');
        }
    }
    if (hEl) hEl.addEventListener('input', calcBMI);
    if (wEl) wEl.addEventListener('input', calcBMI);

    /* ── Sex toggle ── */
    window.selectSex = function(btn) {
        document.querySelectorAll('.sex-btn').forEach(b => b.classList.remove('selected'));
        btn.classList.add('selected');
        const sexInput = document.getElementById('Sex');
        if (sexInput) sexInput.value = btn.dataset.value;
        const g = sexInput?.closest('.form-group') || sexInput?.parentElement;
        if (g) g.classList.remove('has-error');
    };

    /* ── Symptom toggles ── */
    document.querySelectorAll('.symptom-toggle').forEach(t => {
        const sw = t.querySelector('.switch');
        const inp = t.querySelector('input[type=hidden]');
        if (!sw || !inp) return;
        t.addEventListener('click', e => {
            e.preventDefault();
            const on = sw.classList.toggle('on');
            inp.value = on ? 'yes' : 'no';
        });
    });

    /* ── US Performed toggle ── */
    const usSelect = document.getElementById('US_Performed');
    const usFields = document.getElementById('usFields');
    if (usSelect && usFields) {
        function toggleUS() { usFields.classList.toggle('visible', usSelect.value === 'yes'); }
        usSelect.addEventListener('change', toggleUS);
        toggleUS();
    }

    /* ═══ RESULT PAGE ANIMATIONS ═══ */

    /* Ring animation */
    const ringProgress = document.querySelector('.ring-progress');
    const ringPct = document.querySelector('.ring-pct');
    if (ringProgress) {
        const r = 90;
        const c = 2 * Math.PI * r;
        const pct = parseFloat(ringProgress.dataset.pct) || 0;
        ringProgress.style.strokeDasharray = c;
        ringProgress.style.strokeDashoffset = c;
        requestAnimationFrame(() => {
            setTimeout(() => {
                ringProgress.style.transition = 'stroke-dashoffset 1.5s ease-out';
                ringProgress.style.strokeDashoffset = c - (pct / 100) * c;
            }, 300);
        });
        /* Animate number */
        if (ringPct) {
            const target = parseFloat(ringPct.dataset.value) || 0;
            let cur = 0;
            const step = target / 60;
            const timer = setInterval(() => {
                cur += step;
                if (cur >= target) { cur = target; clearInterval(timer); }
                ringPct.textContent = cur.toFixed(1);
            }, 25);
        }
    }

    /* SHAP bar animation */
    document.querySelectorAll('.shap-fill').forEach(bar => {
        const w = bar.dataset.width;
        if (w) {
            setTimeout(() => { bar.style.width = w + '%'; }, 500);
        }
    });

    /* ═══ HISTORY PAGE ═══ */
    window.deleteRecord = function(id) {
        if (!confirm('Supprimer cet enregistrement ?')) return;
        fetch('/history/' + encodeURIComponent(id), { method: 'DELETE' })
            .then(r => { if (r.ok) location.reload(); });
    };

    window.clearHistory = function() {
        if (!confirm('Effacer tout l\'historique ? Cette action est irréversible.')) return;
        fetch('/history/clear', { method: 'POST' })
            .then(r => { if (r.ok) location.reload(); });
    };

    window.filterRecords = function() {
        const q = (document.getElementById('searchInput')?.value || '').toLowerCase();
        const f = document.getElementById('filterResult')?.value || 'all';
        document.querySelectorAll('#historyTable tbody tr').forEach(tr => {
            const text = tr.textContent.toLowerCase();
            const result = tr.dataset.result;
            const matchText = !q || text.includes(q);
            const matchFilter = f === 'all' || result === f;
            tr.style.display = matchText && matchFilter ? '' : 'none';
        });
    };

    /* Auto-dismiss flash */
    document.querySelectorAll('.flash-msg').forEach(m => {
        m.style.transition = 'opacity 0.4s, transform 0.4s';
        setTimeout(() => { m.style.opacity = '0'; m.style.transform = 'translateY(-10px)'; setTimeout(() => m.remove(), 400); }, 3000);
    });
});

/* ═══════════════════════════════════════════════════════════
   PediAppend – diagnosis.js
   Script spécifique au formulaire de diagnostic (diagnosis.html).
   Gère : navigation multi-étapes, validation, calcul de l'âge
   et de l'IMC, sélection du sexe, toggles de symptômes,
   et toggle échographie.
   ═══════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
    const prefersReducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    /* ═══ FORMULAIRE MULTI-ÉTAPES ═══ */
    const steps = document.querySelectorAll('.form-step');
    const nextBtn = document.getElementById('nextBtn');
    const prevBtn = document.getElementById('prevBtn');
    let currentStep = 0;

    /* Affiche l'étape n et met à jour les indicateurs de progression */
    function showStep(n) {
        steps.forEach((s, i) => s.style.display = i === n ? '' : 'none');
        if (prevBtn) prevBtn.style.visibility = n === 0 ? 'hidden' : 'visible';
        if (nextBtn) {
            if (n === steps.length - 1) {
                nextBtn.innerHTML = '&#129657; Analyser &rarr;';
                nextBtn.classList.add('submit-mode');
            } else {
                nextBtn.innerHTML = 'Suivant &rarr;';
                nextBtn.classList.remove('submit-mode');
            }
        }
        /* Cercles d'étape : active / terminée / en attente */
        document.querySelectorAll('.step-circle').forEach((c, i) => {
            c.classList.remove('active', 'completed', 'pending');
            if (i < n) c.classList.add('completed');
            else if (i === n) c.classList.add('active');
            else c.classList.add('pending');
        });
        /* Labels d'étape */
        document.querySelectorAll('.step-label-text').forEach((l, i) => {
            l.classList.toggle('is-active', i <= n);
        });
        /* Barre de progression */
        const fill = document.querySelector('.step-progress-fill');
        if (fill) fill.style.width = ((n + 1) / steps.length * 100) + '%';
        /* Connecteurs entre étapes */
        document.querySelectorAll('.step-connector-fill').forEach((cf, i) => {
            cf.style.width = i < n ? '100%' : '0%';
        });
    }

    /* Valide les champs requis de l'étape n */
    function validateStep(n) {
        const step = steps[n];
        if (!step) return true;
        const required = step.querySelectorAll('[required]');
        let ok = true;
        let firstInvalid = null;
        required.forEach(el => {
            const g = el.closest('.form-group') || el.closest('.bmi-card') || el.parentElement;
            if (!el.value) {
                g.classList.add('has-error');
                ok = false;
                if (!firstInvalid && typeof el.focus === 'function' && el.type !== 'hidden') firstInvalid = el;
            } else {
                g.classList.remove('has-error');
            }
        });
        /* Validation spéciale pour le champ Sexe (input hidden) */
        const sexInput = step.querySelector('#Sex');
        if (sexInput && !sexInput.value) {
            const g = sexInput.closest('.form-group') || sexInput.parentElement;
            g.classList.add('has-error');
            ok = false;
            if (!firstInvalid) firstInvalid = step.querySelector('.sex-btn') || sexInput;
        }
        if (!ok && firstInvalid && typeof firstInvalid.scrollIntoView === 'function') {
            firstInvalid.scrollIntoView({ behavior: prefersReducedMotion ? 'auto' : 'smooth', block: 'center' });
            if (typeof firstInvalid.focus === 'function') firstInvalid.focus({ preventScroll: true });
        }
        return ok;
    }

    /* Bouton Suivant : valide l'étape courante puis avance ou soumet */
    if (nextBtn) {
        nextBtn.addEventListener('click', () => {
            if (!validateStep(currentStep)) return;
            if (currentStep === steps.length - 1) {
                document.getElementById('diagnosisForm').submit();
            } else {
                currentStep++;
                showStep(currentStep);
                window.scrollTo({ top: document.querySelector('.form-section').offsetTop - 100, behavior: prefersReducedMotion ? 'auto' : 'smooth' });
            }
        });
    }
    /* Bouton Précédent : recule d'une étape */
    if (prevBtn) {
        prevBtn.addEventListener('click', () => {
            if (currentStep > 0) { currentStep--; showStep(currentStep); }
        });
    }
    /* Initialisation : affiche la première étape */
    if (steps.length) showStep(0);

    /* ═══ CALCUL DE L'ÂGE ═══ */
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

    /* ═══ CALCUL DE L'IMC ═══ */
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

    /* ═══ SÉLECTION DU SEXE ═══ */
    document.querySelectorAll('.sex-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.sex-btn').forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');
            const sexInput = document.getElementById('Sex');
            if (sexInput) sexInput.value = btn.dataset.value;
            const g = sexInput?.closest('.form-group') || sexInput?.parentElement;
            if (g) g.classList.remove('has-error');
        });
    });

    /* ═══ TOGGLES DE SYMPTÔMES ═══ */
    /* Chaque toggle bascule entre 'yes' et 'no' dans un champ hidden */
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

    /* ═══ TOGGLE ÉCHOGRAPHIE ═══ */
    /* Affiche/masque les champs d'échographie selon la sélection */
    const usSelect = document.getElementById('US_Performed');
    const usFields = document.getElementById('usFields');
    if (usSelect && usFields) {
        function toggleUS() { usFields.classList.toggle('visible', usSelect.value === 'yes'); }
        usSelect.addEventListener('change', toggleUS);
        toggleUS();
    }

});

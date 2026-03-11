/* ═══════════════════════════════════════════════════════════
   PediAppend – common.js
   Code commun chargé sur toutes les pages via base.html.
   Gère : navbar scroll, smooth scroll, animations d'entrée,
   et auto-dismiss des messages flash.
   ═══════════════════════════════════════════════════════════ */
/** Seuil de défilement (px) au-delà duquel la navbar passe en mode 'scrolled'. */
const NAVBAR_SCROLL_THRESHOLD = 20;

/** Seuil de visibilité (0–1) pour l'IntersectionObserver des animations d'entrée. */
const INTERSECTION_THRESHOLD = 0.08;

document.addEventListener('DOMContentLoaded', () => {
    const prefersReducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    /* ── Navbar : ajout de la classe 'scrolled' au-delà de NAVBAR_SCROLL_THRESHOLD ── */
    const nav = document.querySelector('.navbar');
    if (nav) {
        window.addEventListener('scroll', () => {
            nav.classList.toggle('scrolled', window.scrollY > NAVBAR_SCROLL_THRESHOLD);
        });
        nav.classList.toggle('scrolled', window.scrollY > NAVBAR_SCROLL_THRESHOLD);
    }

    /* ── Smooth scroll pour les ancres (#) sauf les liens de navigation de slides ── */
    document.querySelectorAll('a[href^="#"]').forEach(a => {
        a.addEventListener('click', e => {
            if (a.hasAttribute('data-goto')) return;
            const t = document.querySelector(a.getAttribute('href'));
            if (t) { e.preventDefault(); t.scrollIntoView({ behavior: prefersReducedMotion ? 'auto' : 'smooth', block: 'start' }); }
        });
    });

    /* ── IntersectionObserver : ajoute la classe 'visible' aux éléments .animate-in ── */
    if (prefersReducedMotion) {
        document.querySelectorAll('.animate-in').forEach(el => el.classList.add('visible'));
    } else {
        const io = new IntersectionObserver(entries => {
            entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('visible'); io.unobserve(e.target); } });
        }, { threshold: INTERSECTION_THRESHOLD });
        document.querySelectorAll('.animate-in').forEach(el => io.observe(el));
    }

    /* ── Auto-dismiss des messages flash après 3 secondes ── */
    document.querySelectorAll('.flash-msg').forEach(m => {
        if (prefersReducedMotion) {
            setTimeout(() => m.remove(), 3000);
        } else {
            m.style.transition = 'opacity 0.4s, transform 0.4s';
            setTimeout(() => { m.style.opacity = '0'; m.style.transform = 'translateY(-10px)'; setTimeout(() => m.remove(), 400); }, 3000);
        }
    });

});

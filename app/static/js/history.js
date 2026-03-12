/* ═══════════════════════════════════════════════════════════
   PediAppend – history.js
   Script spécifique à la page d'historique (history.html).
   Gère : suppression d'un enregistrement, effacement complet
   de l'historique, et filtrage côté client.
   ═══════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {

    /* ═══ FILTRAGE CÔTÉ CLIENT ═══ */
    /* Filtre les lignes du tableau par texte de recherche et type de résultat */
    function filterRecords() {
        const q = (document.getElementById('searchInput')?.value || '').toLowerCase();
        const f = document.getElementById('filterResult')?.value || 'all';
        let anyVisible = false;
        document.querySelectorAll('#historyTable tbody tr').forEach(tr => {
            const text = tr.textContent.toLowerCase();
            const result = tr.dataset.result;
            const matchText = !q || text.includes(q);
            const matchFilter = f === 'all' || result === f;
            const show = matchText && matchFilter;
            tr.style.display = show ? '' : 'none';
            if (show) anyVisible = true;
        });
        const empty = document.getElementById('filterEmpty');
        if (empty) empty.hidden = anyVisible;
    }

    const table = document.getElementById('historyTable');
    if (table) {
        table.addEventListener('click', (e) => {
            const btn = e.target.closest('button[data-action]');
            if (!btn) return;
            const action = btn.dataset.action;
            if (action !== 'delete-record') return;
            const id = btn.dataset.recordId;
            if (!id) return;
            if (!confirm('Supprimer cet enregistrement ?')) return;
            fetch('/history/' + encodeURIComponent(id), { method: 'DELETE' })
                .then(r => { if (r.ok) location.reload(); })
                .catch(err => console.error('[history] Erreur suppression :', err));
        });
    }

    const clearBtn = document.querySelector('button[data-action="clear-history"]');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            if (!confirm('Effacer tout l\'historique ? Cette action est irréversible.')) return;
            fetch('/history/clear', { method: 'POST' })
                .then(r => { if (r.ok) location.reload(); })
                .catch(err => console.error('[history] Erreur effacement :', err));
        });
    }

    const search = document.getElementById('searchInput');
    const select = document.getElementById('filterResult');
    if (search) search.addEventListener('input', filterRecords);
    if (select) select.addEventListener('change', filterRecords);

    // Initialize empty-state for filters on first render.
    if (document.getElementById('historyTable')) filterRecords();

});

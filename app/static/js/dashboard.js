/* -------------------- Modal Control -------------------- */
function openModal(id) {
    document.getElementById(id).classList.add('show');
}

function closeModal(id) {
    document.getElementById(id).classList.remove('show');
}

/* -------------------- AI Matching -------------------- */
async function fetchAIMatching() {
    const contentDiv = document.getElementById('aiMatchingContent');
    contentDiv.innerHTML = '<div class="loading">Fetching top career matches…</div>';

    try {
        const resp = await fetch('/dashboard/ai_matching');
        const data = await resp.json();

        if (data.error) throw new Error(data.error);

        const html = data.map(c => `
            <div class="career-card" onclick="openCareerPath('${c.title.replace(/'/g, "\\'")}')">
                <h4>${c.title}</h4>
                <div class="score">Match Score: ${c.match_score}%</div>
            </div>
        `).join('');

        contentDiv.innerHTML = html || '<p>No matches found.</p>';
    } catch (e) {
        contentDiv.innerHTML = `<p style="color:red;">Error: ${e.message}</p>`;
    }
}

async function openAIMatching() {
    openModal('aiMatchingModal');
    await fetchAIMatching();
}

/* -------------------- Career Path Explorer -------------------- */
async function openCareerPathExplorer() {
    openModal('careerExplorerModal');
    const contentDiv = document.getElementById('careerExplorerContent');
    contentDiv.innerHTML = '<div class="loading">Fetching all career paths…</div>';

    try {
        const resp = await fetch('/dashboard/career_path');
        const data = await resp.json();

        if (data.error) throw new Error(data.error);

        const html = data.map(c => `
            <div class="career-card" onclick="openCareerPath('${c.title.replace(/'/g, "\\'")}')">
                <h4>${c.title}</h4>
                <div class="score">Match Score: ${c.match_score}%</div>
                <p class="rationale"><strong>Why:</strong> ${c.rationale}</p>
                <p><strong>Roadmap:</strong> ${c.roadmap}</p>
                <p><strong>Timeline:</strong> ${c.timeline}</p>
            </div>
        `).join('');

        contentDiv.innerHTML = html || '<p>No careers found.</p>';
    } catch (e) {
        contentDiv.innerHTML = `<p style="color:red;">Error: ${e.message}</p>`;
    }
}

/* -------------------- Detailed Career Path Modal -------------------- */
async function openCareerPath(title) {
    closeModal('aiMatchingModal');
    closeModal('careerExplorerModal');
    openModal('careerModal');

    const contentDiv = document.getElementById('careerPathContent');
    contentDiv.innerHTML = '<div class="loading">Fetching detailed roadmap…</div>';

    try {
        const resp = await fetch(`/dashboard/career_path?title=${encodeURIComponent(title)}`);
        const data = await resp.json();

        if (data.error) throw new Error(data.error);

        contentDiv.innerHTML = `
            <h4>${data.title}</h4>
            <p><strong>Match Score:</strong> ${data.match_score}%</p>
            <p><strong>Roadmap:</strong> ${data.roadmap}</p>
            <p><strong>Timeline:</strong> ${data.timeline}</p>
            <p><strong>Skills Needed:</strong> ${data.skills}</p>
            <p><strong>Why:</strong> ${data.rationale}</p>
            <p><em>Click outside or press ESC to close.</em></p>
        `;
    } catch (e) {
        contentDiv.innerHTML = `<p style="color:red;">Error: ${e.message}</p>`;
    }
}

/* -------------------- Placeholder Functions -------------------- */
function openResources() { 
    alert('Learning Resources feature coming soon!'); 
}

function openQuiz() { 
    alert('Interest & Skills Quiz coming soon!'); 
}

/* -------------------- Close Modal on Background or Escape -------------------- */
document.querySelectorAll('.modal').forEach(modal => {
    modal.addEventListener('click', e => {
        if (e.target === modal) closeModal(modal.id);
    });
});

document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
        document.querySelectorAll('.modal.show').forEach(m => closeModal(m.id));
    }
});

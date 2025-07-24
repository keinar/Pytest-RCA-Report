function bugToString(bug) {
    if (typeof bug === 'string') return bug;
    if (typeof bug === 'object') {
        if (bug.name && bug.reason)
            return `${bug.name}: ${bug.reason}`;
        if (bug.name)
            return bug.name;
        if (bug.reason)
            return bug.reason;
        return JSON.stringify(bug);
    }
    return String(bug);
}

window.addEventListener('DOMContentLoaded', () => {
    fetch('results.json')
        .then(resp => {
            if (!resp.ok) throw new Error('HTTP ' + resp.status);
            return resp.json();
        })
        .then(data => {
            // 1) Results Summary
            document.getElementById('summary-table').innerHTML = `
        <table>
          <tr class="passed"><td>Passed</td><td>${data.summary.passed}</td></tr>
          <tr class="failed"><td>Failed</td><td>${data.summary.failed}</td></tr>
          <tr class="skipped"><td>Skipped</td><td>${data.summary.skipped}</td></tr>
        </table>
      `;

            // 2) Pie/Bar graphs (Chart.js)
            const ctxPie = document.getElementById('summary-pie').getContext('2d');
            const ctxBar = document.getElementById('summary-bar').getContext('2d');
            const d = data.summary;
            new Chart(ctxPie, {
                type: 'pie',
                data: {
                    labels: ['Passed','Failed','Skipped'],
                    datasets: [{
                        data: [d.passed,d.failed,d.skipped],
                        backgroundColor: ['#2ecc40','#e74c3c','#f9a825']
                    }]
                }
            });
            new Chart(ctxBar, {
                type: 'bar',
                data: {
                    labels: ['Passed','Failed','Skipped'],
                    datasets: [{
                        label: 'Count',
                        data: [d.passed,d.failed,d.skipped],
                        backgroundColor: ['#2ecc40','#e74c3c','#f9a825']
                    }]
                }
            });

            // 3) Log Anomalies
            const anomaliesSection = document.getElementById('anomalies');
            const listEl = document.getElementById('anomaly-list');
            listEl.innerHTML = '';
            function renderAnomalyGroup(title, entries) {
                if (!entries || Object.keys(entries).length === 0) return;
                const subH3 = document.createElement('h3');
                subH3.textContent = title;
                anomaliesSection.appendChild(subH3);
                Object.entries(entries).forEach(([message, count]) => {
                    const li = document.createElement('li');
                    if (count !== null && typeof count === 'object') {
                        const details = Object.entries(count)
                            .map(([k, v]) => `${k}: ${v}`)
                            .join(', ');
                        li.textContent = `${message} (${details})`;
                    } else {
                        li.textContent = `${message} (${count})`;
                    }
                    listEl.appendChild(li);
                });
            }
            const anomalies = data.anomalies || {};
            const errorGroup = anomalies.errors || anomalies.recurring_errors;
            const warningGroup = anomalies.warnings;
            renderAnomalyGroup('Errors', errorGroup);
            renderAnomalyGroup('Warnings', warningGroup);
            if (listEl.childElementCount === 0) {
                anomaliesSection.style.display = 'none';
            }

            // 4) Root Cause Analysis
            const tbody = document.querySelector('#rca-table tbody');
            Object.entries(data.root_cause || {}).forEach(([mod, cnt]) => {
                const tr = document.createElement('tr');
                let display = '';
                if (cnt !== null && typeof cnt === 'object') {
                    display = Object.entries(cnt)
                        .map(([k, v]) => `${k}: ${v}`)
                        .join(', ');
                } else {
                    display = cnt;
                }
                tr.innerHTML = `<td>${mod}</td><td>${display}</td>`;
                tbody.appendChild(tr);
            });

            // 5) Execution Time Distribution
            const visualsSection = document.getElementById('visuals');
            const img = document.getElementById('time-dist');
            if (data.chart_time_dist) {
                img.src = data.chart_time_dist;
            } else {
                visualsSection.style.display = 'none';
            }

            // 6) Slowest Tests
            const slowList = document.getElementById('slow-tests');
            if (data.slowest_tests && data.slowest_tests.length) {
                data.slowest_tests.forEach(name => {
                    const li = document.createElement('li');
                    li.textContent = name;
                    slowList.appendChild(li);
                });
            } else {
                slowList.previousElementSibling.style.display = 'none';
            }

            // 7) Actionable Recommendations
            const recList = document.getElementById('rec-list');
            (data.recommendations || []).forEach(r => {
                const li = document.createElement('li');
                li.textContent = r;
                recList.appendChild(li);
            });

            // 8) Failure Classification
            if (data.failure_classification) {
                const section = document.getElementById('failure-classification');
                section.style.display = '';
                section.innerHTML = `
                  <h2>Failure Classification</h2>
                  <h3>Likely Real Bugs</h3>
                  <ul>${(data.failure_classification.real_bugs || []).map(bugToString).map(txt => `<li>${txt}</li>`).join('')}</ul>
                  <h3>Likely Test Issues</h3>
                  <ul>${(data.failure_classification.test_issues || []).map(bugToString).map(txt => `<li>${txt}</li>`).join('')}</ul>
                `;
            }

            // 9) All testcases table with search/filter/status/drilldown
            const testTableBody = document.querySelector('#test-table tbody');
            const searchInput = document.getElementById('test-search');
            const statusFilterDiv = document.getElementById('status-filter');
            let currentStatusFilter = "";

            function renderTable(filter = "", statusFilter = "") {
                testTableBody.innerHTML = "";
                (data.testcases || []).forEach(tc => {
                    if (!tc.name.toLowerCase().includes(filter.toLowerCase())) return;
                    if (statusFilter && tc.status !== statusFilter) return;
                    const tr = document.createElement('tr');
                    tr.className = tc.status;
                    tr.innerHTML = `<td class="test-name">${tc.name}</td>
                        <td>${tc.status}</td>
                        <td>${Object.keys(tc.properties).map(k => `${k}: ${tc.properties[k]}`).join(", ")}</td>`;
                    tr.addEventListener('click', () => {
                        showTestDetails(tc);
                    });
                    testTableBody.appendChild(tr);
                });
            }
            searchInput.addEventListener('input', e => renderTable(e.target.value, currentStatusFilter));
            statusFilterDiv.querySelectorAll('button').forEach(btn => {
                btn.addEventListener('click', e => {
                    currentStatusFilter = btn.dataset.status;
                    renderTable(searchInput.value, currentStatusFilter);
                    statusFilterDiv.querySelectorAll('button').forEach(b=>b.classList.remove('active'));
                    btn.classList.add('active');
                });
            });
            statusFilterDiv.querySelector('button[data-status=""]').classList.add('active'); // הדגש "All" כברירת מחדל

            renderTable();

            // Drilldown modal
            window.showTestDetails = function(tc) {
                let explanation = '';
                if (data.failure_classification) {
                    const all = [...(data.failure_classification.real_bugs||[]), ...(data.failure_classification.test_issues||[])];
                    const found = all.find(b => typeof b === 'object' && b.name === tc.name);
                    if (found) explanation = found.reason || '';
                }
                const html = `
                  <div style="padding:24px;">
                    <h3>${tc.name}</h3>
                    <p><strong>Status:</strong> ${tc.status}</p>
                    <p><strong>Properties:</strong> ${Object.keys(tc.properties).map(k => `${k}: ${tc.properties[k]}`).join(", ") || '(none)'}</p>
                    ${explanation ? `<p><strong>Failure Reason:</strong> ${explanation}</p>` : ''}
                    <button onclick="document.getElementById('test-detail-modal').style.display='none'">Close</button>
                  </div>
                `;
                let modal = document.getElementById('test-detail-modal');
                if (!modal) {
                    modal = document.createElement('div');
                    modal.id = 'test-detail-modal';
                    modal.style.position = 'fixed';
                    modal.style.top = 0; modal.style.left = 0; modal.style.right = 0; modal.style.bottom = 0;
                    modal.style.background = 'rgba(0,0,0,0.35)';
                    modal.style.zIndex = 9999;
                    modal.style.display = 'none';
                    document.body.appendChild(modal);
                }
                modal.innerHTML = `<div style="background:#fff;max-width:600px;margin:80px auto;border-radius:10px;box-shadow:0 4px 16px #0002;">${html}</div>`;
                modal.style.display = '';
            };

            // Download JSON
            document.getElementById('download-json').onclick = function() {
                fetch('results.json')
                    .then(r=>r.blob())
                    .then(blob=>{
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url; a.download = 'results.json';
                        a.click();
                        URL.revokeObjectURL(url);
                    });
            };
            // Download CSV
            document.getElementById('download-csv').onclick = function() {
                let csv = "Name,Status,Properties\n";
                (data.testcases||[]).forEach(tc=>{
                    csv += `"${tc.name}","${tc.status}","${Object.keys(tc.properties).map(k=>`${k}: ${tc.properties[k]}`).join("; ")}"\n`;
                });
                const blob = new Blob([csv], {type:'text/csv'});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url; a.download = 'results.csv';
                a.click();
                URL.revokeObjectURL(url);
            };

        })
        .catch(err => {
            console.error('Failed to load or parse results.json:', err);
        });
});

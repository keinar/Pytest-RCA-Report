function fetchHistoryRuns() {
    return fetch('data/history/')
        .then(resp => {
            if (!resp.ok) throw new Error('HTTP ' + resp.status);
            return resp.text();
        })
        .then(html => {
            // Extract JSON filenames from directory listing HTML
            const parser = new DOMParser();
            const doc = parser.parseFromString(html, 'text/html');
            const links = Array.from(doc.querySelectorAll('a'));
            const jsonFiles = links.map(a => a.getAttribute('href')).filter(href => href && href.endsWith('.json'));
            return Promise.all(jsonFiles.map(file => fetch('data/history/' + file).then(r => r.json()).then(data => ({file, data}))));
        })
        .then(filesData => {
            // Sort by filename (assuming filename includes date or chronological order)
            filesData.sort((a, b) => a.file.localeCompare(b.file));
            return filesData.map(fd => fd.data);
        })
        .catch(() => []); // If error (e.g. no directory listing), return empty array
}

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
    // Helper: render dashboard sections given data and historyRuns (if available)
    async function renderDashboard(data, historyRuns, selectedRunIdx) {
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
        // Destroy previous charts if they exist (avoid Chart.js error)
        if (window._summaryPieChart) window._summaryPieChart.destroy();
        if (window._summaryBarChart) window._summaryBarChart.destroy();
        window._summaryPieChart = new Chart(ctxPie, {
            type: 'pie',
            data: {
                labels: ['Passed','Failed','Skipped'],
                datasets: [{
                    data: [d.passed,d.failed,d.skipped],
                    backgroundColor: ['#2ecc40','#e74c3c','#f9a825']
                }]
            }
        });
        window._summaryBarChart = new Chart(ctxBar, {
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
        } else {
            anomaliesSection.style.display = '';
        }

        // 4) Root Cause Analysis
        const tbody = document.querySelector('#rca-table tbody');
        tbody.innerHTML = '';
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
            visualsSection.style.display = '';
        } else {
            visualsSection.style.display = 'none';
        }

        // 6) Slowest Tests
        const slowList = document.getElementById('slow-tests');
        slowList.innerHTML = '';
        if (data.slowest_tests && data.slowest_tests.length) {
            data.slowest_tests.forEach(name => {
                const li = document.createElement('li');
                li.textContent = name;
                slowList.appendChild(li);
            });
            slowList.previousElementSibling.style.display = '';
        } else {
            if (slowList.previousElementSibling)
                slowList.previousElementSibling.style.display = 'none';
        }

        // 7) Actionable Recommendations
        const recList = document.getElementById('rec-list');
        recList.innerHTML = '';
        (Array.isArray(data.recommendations) ? data.recommendations : []).forEach(r => {
            const li = document.createElement('li');
            li.textContent = r;
            recList.appendChild(li);
        });

        // 8) Failure Classification
        const section = document.getElementById('failure-classification');
        if (data.failure_classification) {
            section.style.display = '';
            section.innerHTML = `
              <h2>Failure Classification</h2>
              <h3>Likely Real Bugs</h3>
              <ul>${(data.failure_classification.real_bugs || []).map(bugToString).map(txt => `<li>${txt}</li>`).join('')}</ul>
              <h3>Likely Test Issues</h3>
              <ul>${(data.failure_classification.test_issues || []).map(bugToString).map(txt => `<li>${txt}</li>`).join('')}</ul>
            `;
        } else {
            section.style.display = 'none';
        }

        // 9) Regression Trends Section and Chart
        const regressionSection = document.getElementById('regression-trends');
        if (historyRuns && historyRuns.length >= 2) {
            regressionSection.style.display = '';
            // Prepare data for line chart
            const labels = historyRuns.map((run, idx) => {
                if (run.data.date) return run.data.date;
                if (run.data.run_date) return run.data.run_date;
                return run.file || `Run ${idx+1}`;
            });
            const passedData = historyRuns.map(run => run.data.summary?.passed || 0);
            const failedData = historyRuns.map(run => run.data.summary?.failed || 0);
            const skippedData = historyRuns.map(run => run.data.summary?.skipped || 0);

            const ctxLine = document.getElementById('regression-chart').getContext('2d');
            if (window._regressionLineChart) window._regressionLineChart.destroy();
            window._regressionLineChart = new Chart(ctxLine, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Passed',
                            data: passedData,
                            borderColor: '#2ecc40',
                            backgroundColor: '#2ecc40',
                            fill: false,
                            tension: 0.1
                        },
                        {
                            label: 'Failed',
                            data: failedData,
                            borderColor: '#e74c3c',
                            backgroundColor: '#e74c3c',
                            fill: false,
                            tension: 0.1
                        },
                        {
                            label: 'Skipped',
                            data: skippedData,
                            borderColor: '#f9a825',
                            backgroundColor: '#f9a825',
                            fill: false,
                            tension: 0.1
                        }
                    ]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'top',
                        },
                        title: {
                            display: true,
                            text: 'Regression Trends'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            precision: 0
                        }
                    }
                }
            });
        } else {
            regressionSection.style.display = 'none';
        }

        // 10) All testcases table with search/filter/status/drilldown
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
        // Remove previous event listeners (on rerender)
        const newSearchInput = searchInput.cloneNode(true);
        searchInput.parentNode.replaceChild(newSearchInput, searchInput);
        newSearchInput.addEventListener('input', e => renderTable(e.target.value, currentStatusFilter));
        const newStatusFilterDiv = statusFilterDiv.cloneNode(true);
        statusFilterDiv.parentNode.replaceChild(newStatusFilterDiv, statusFilterDiv);
        newStatusFilterDiv.querySelectorAll('button').forEach(btn => {
            btn.addEventListener('click', e => {
                currentStatusFilter = btn.dataset.status;
                renderTable(newSearchInput.value, currentStatusFilter);
                newStatusFilterDiv.querySelectorAll('button').forEach(b=>b.classList.remove('active'));
                btn.classList.add('active');
            });
        });
        newStatusFilterDiv.querySelector('button[data-status=""]').classList.add('active');
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
            // If viewing a history run, download that file, else results.json
            let url = 'results.json';
            if (historyRuns && selectedRunIdx != null && historyRuns[selectedRunIdx] && historyRuns[selectedRunIdx].file) {
                url = 'data/history/' + historyRuns[selectedRunIdx].file;
            }
            fetch(url)
                .then(r=>r.blob())
                .then(blob=>{
                    const objUrl = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = objUrl; a.download = url.split('/').pop();
                    a.click();
                    URL.revokeObjectURL(objUrl);
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
    }

    // Start main logic as async IIFE
    (async () => {
        try {
            const resp = await fetch('results.json');
            if (!resp.ok) throw new Error('HTTP ' + resp.status);
            const data = await resp.json();

            // Load history runs for regression trends chart and history list
            const historyRunsRaw = await fetch('data/history/')
                .then(resp => {
                    if (!resp.ok) throw new Error('HTTP ' + resp.status);
                    return resp.text();
                })
                .then(html => {
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(html, 'text/html');
                    const links = Array.from(doc.querySelectorAll('a'));
                    const jsonFiles = links.map(a => a.getAttribute('href')).filter(href => href && href.endsWith('.json'));
                    return Promise.all(jsonFiles.map(file => fetch('data/history/' + file).then(r => r.json()).then(data => ({file, data}))));
                })
                .then(filesData => {
                    filesData.sort((a, b) => a.file.localeCompare(b.file));
                    return filesData;
                })
                .catch(() => []);

            let historyRuns = historyRunsRaw.slice();
            if (!historyRuns.length || (historyRuns.length && historyRuns[historyRuns.length-1].file !== 'results.json')) {
                historyRuns.push({file: 'results.json', data});
            }

            let selectedRunIdx = historyRuns.length - 1;
            let selectedData = historyRuns[selectedRunIdx].data;

            // Render the dashboard for the latest run
            await renderDashboard(selectedData, historyRuns, selectedRunIdx);

            // Render the history list
            const historyListDiv = document.getElementById('history-list');
            function readableDate(run) {
                return run.data.date || run.data.run_date || run.file.replace(/^results_/, '').replace(/\.json$/,'') || run.file;
            }
            function summaryCounts(run) {
                const s = run.data.summary || {};
                let parts = [];

                // רגילים
                if ('regular_passed' in s || 'regular_failed' in s || 'regular_skipped' in s) {
                    parts.push(`R:P:${s.regular_passed ?? 0} F:${s.regular_failed ?? 0} S:${s.regular_skipped ?? 0}`);
                } else if ('passed' in s || 'failed' in s || 'skipped' in s) {
                    // פורמט ישן
                    parts.push(`R:P:${s.passed ?? 0} F:${s.failed ?? 0} S:${s.skipped ?? 0}`);
                }

                // שליליים
                if ('negative_passed' in s || 'negative_failed' in s || 'negative_skipped' in s) {
                    let np = (typeof s.negative_passed === "number") ? `P:${s.negative_passed}` : '';
                    let nf = (typeof s.negative_failed === "number") ? `F:${s.negative_failed}` : '';
                    let ns = (typeof s.negative_skipped === "number") ? `S:${s.negative_skipped}` : '';
                    let neg = ['N', np, nf, ns].filter(Boolean).join(':');
                    if (neg.length > 1) parts.push(neg);
                }

                return `[${parts.join(' ')}]`;
            }
            function renderHistoryList() {
                historyListDiv.innerHTML = '';
                if (!historyRuns.length) {
                    historyListDiv.innerHTML = '<em>No history found.</em>';
                    return;
                }
                const ul = document.createElement('ul');
                ul.style.listStyle = 'none';
                ul.style.padding = '0';
                historyRuns.forEach((run, idx) => {
                    const li = document.createElement('li');
                    li.style.cursor = 'pointer';
                    li.style.padding = '4px 8px';
                    li.style.marginBottom = '2px';
                    li.style.borderRadius = '6px';
                    li.style.transition = 'background 0.2s';
                    if (idx === selectedRunIdx) {
                        li.style.background = '#e0ecff';
                        li.style.fontWeight = 'bold';
                    }
                    li.innerHTML = `<span style="color:#888;">${idx+1}.</span> <span>${readableDate(run)}</span> <span style="font-size:90%;color:#555;">[${summaryCounts(run)}]</span>`;
                    li.addEventListener('click', async () => {
                        if (selectedRunIdx === idx) return;
                        selectedRunIdx = idx;
                        await renderDashboard(historyRuns[selectedRunIdx].data, historyRuns, selectedRunIdx);
                        renderHistoryList();
                    });
                    ul.appendChild(li);
                });
                historyListDiv.appendChild(ul);
            }
            renderHistoryList();
        } catch (err) {
            console.error('Failed to load or parse results.json:', err);
        }
    })();
});

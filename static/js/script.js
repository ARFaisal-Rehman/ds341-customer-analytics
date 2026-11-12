document.addEventListener('DOMContentLoaded', function() {
    // Toggle Sidebar
    const sidebarToggle = document.body.querySelector('#sidebarToggle');
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', event => {
            event.preventDefault();
            document.body.classList.toggle('sb-sidenav-toggled');
        });
    }

    // Page Specific Logic
    const path = window.location.pathname;

    if (path === '/' || path === '/overview') {
        loadOverviewData();
    } else if (path === '/models') {
        loadModelData();
    } else if (path === '/segments') {
        loadSegmentData();
    } else if (path === '/recommendations') {
        loadRulesData();
    }
});

// Helper: Format Number
const formatNum = (num) => new Intl.NumberFormat().format(num);

// 1. Overview Page Logic
function loadOverviewData() {
    // Fetch Stats
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            if (data.error) return;
            const totalEl = document.getElementById('total-customers');
            const churnEl = document.getElementById('churn-rate');
            
            if (totalEl) totalEl.innerText = formatNum(data.total_customers);
            if (churnEl) churnEl.innerText = data.churn_rate + '%';
            
            // Render Churn Chart
            const ctx = document.getElementById('churnChart');
            if (ctx) {
                new Chart(ctx.getContext('2d'), {
                    type: 'doughnut',
                    data: {
                        labels: ['Retained', 'Churned'],
                        datasets: [{
                            data: [data.churn_counts['0'], data.churn_counts['1']],
                            backgroundColor: ['#4e73df', '#e74a3b'],
                            borderWidth: 0,
                            hoverOffset: 10
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        cutout: '75%',
                        plugins: {
                            legend: {
                                position: 'bottom',
                                labels: { usePointStyle: true }
                            }
                        },
                        animation: {
                            animateScale: true,
                            animateRotate: true
                        }
                    }
                });
            }
        });
        
    // Fetch top accuracy for quick stat
    fetch('/api/metrics')
        .then(response => response.json())
        .then(data => {
            const accEl = document.getElementById('top-accuracy');
            if (accEl) {
                const maxAcc = Math.max(...Object.values(data).map(m => m.Accuracy));
                accEl.innerText = (maxAcc * 100).toFixed(1) + '%';
            }
        });
}

// 2. Models Page Logic
function loadModelData() {
    fetch('/api/metrics')
        .then(response => response.json())
        .then(data => {
            const models = Object.keys(data);
            const accuracy = models.map(m => m.Accuracy);
            // Fix: Map correctly
            const accValues = models.map(m => data[m].Accuracy);
            const f1Values = models.map(m => data[m].F1);
            const precValues = models.map(m => data[m].Precision);
            const recValues = models.map(m => data[m].Recall);

            const ctx = document.getElementById('modelChart');
            if (ctx) {
                new Chart(ctx.getContext('2d'), {
                    type: 'bar',
                    data: {
                        labels: models,
                        datasets: [
                            {
                                label: 'Accuracy',
                                data: accValues,
                                backgroundColor: '#4e73df'
                            },
                            {
                                label: 'F1 Score',
                                data: f1Values,
                                backgroundColor: '#f6c23e'
                            },
                            {
                                label: 'Precision',
                                data: precValues,
                                backgroundColor: '#1cc88a'
                            },
                            {
                                label: 'Recall',
                                data: recValues,
                                backgroundColor: '#36b9cc'
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 1.0,
                                grid: { color: '#f8f9fc' }
                            },
                            x: {
                                grid: { display: false }
                            }
                        },
                        plugins: {
                            legend: { position: 'top' },
                            tooltip: {
                                mode: 'index',
                                intersect: false,
                                backgroundColor: 'rgba(255, 255, 255, 0.9)',
                                titleColor: '#6e707e',
                                bodyColor: '#858796',
                                borderColor: '#dddfeb',
                                borderWidth: 1
                            }
                        }
                    }
                });
            }
        });
}

// 3. Segments Page Logic
function loadSegmentData() {
    fetch('/api/clusters')
        .then(response => response.json())
        .then(data => {
            const tableBody = document.querySelector('#cluster-table tbody');
            if (!tableBody) return;
            
            tableBody.innerHTML = ''; // Clear loading
            
            data.forEach(row => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><span class="badge bg-secondary">Cluster ${row.Cluster}</span></td>
                    <td>${Math.round(row.Recency)}</td>
                    <td>${Math.round(row.Frequency)}</td>
                    <td><strong>$${formatNum(Math.round(row.Monetary))}</strong></td>
                `;
                tableBody.appendChild(tr);
            });
        });
}

// 4. Recommendations Page Logic
function loadRulesData() {
    fetch('/api/rules')
        .then(response => response.json())
        .then(data => {
            const tableBody = document.querySelector('#rules-table tbody');
            if (!tableBody) return;
            
            tableBody.innerHTML = ''; // Clear loading
            
            // Take top 10
            data.slice(0, 10).forEach(row => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="text-primary fw-bold">${row.antecedents.join(', ')}</td>
                    <td class="text-success fw-bold"><i class="fas fa-arrow-right me-2 text-muted"></i> ${row.consequents.join(', ')}</td>
                    <td>
                        <div class="progress" style="height: 20px;">
                            <div class="progress-bar bg-info" role="progressbar" style="width: ${row.confidence * 100}%" 
                                aria-valuenow="${row.confidence * 100}" aria-valuemin="0" aria-valuemax="100">
                                ${Math.round(row.confidence * 100)}%
                            </div>
                        </div>
                    </td>
                `;
                tableBody.appendChild(tr);
            });
        });
}

let uploadedFile = null;
let resultsData = null;

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');

uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

function handleFile(file) {
    if (!file.name.endsWith('.csv')) { alert('Please upload a CSV file'); return; }
    uploadedFile = file;
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileSize').textContent = formatFileSize(file.size);
    document.getElementById('fileInfo').classList.remove('hidden');
    document.getElementById('configPanel').classList.remove('hidden');
    document.getElementById('resultsSection').classList.add('hidden');
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

async function runOptimization() {
    if (!uploadedFile) { alert('Please upload a file first'); return; }

    document.getElementById('loadingSection').classList.remove('hidden');
    document.getElementById('resultsSection').classList.add('hidden');

    const formData = new FormData();
    formData.append('file', uploadedFile);
    formData.append('optimization_goal', document.getElementById('optimizationGoal').value);
    formData.append('vehicle_capacity', document.getElementById('vehicleCapacity').value);
    formData.append('max_distance', document.getElementById('maxDistance').value);

    try {
        const response = await fetch('/api/optimize-logistics', { method: 'POST', body: formData });
        const data = await response.json();

        if (data.error) { alert('Error: ' + data.error); document.getElementById('loadingSection').classList.add('hidden'); return; }

        resultsData = data;
        displayResults(data);
        document.getElementById('loadingSection').classList.add('hidden');
        document.getElementById('resultsSection').classList.remove('hidden');
        document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
    } catch (error) {
        alert('Error: ' + error.message);
        document.getElementById('loadingSection').classList.add('hidden');
    }
}

function displayResults(data) {
    document.getElementById('costSavings').textContent = data.improvements.cost_savings.toFixed(1) + '%';
    document.getElementById('costAmount').textContent = 'Significant savings';
    document.getElementById('distanceReduction').textContent = data.improvements.distance_reduction.toFixed(1) + '%';
    document.getElementById('delayReduction').textContent = data.improvements.delay_reduction.toFixed(0) + '%';
    document.getElementById('efficiency').textContent = '+' + data.improvements.efficiency_improvement.toFixed(1) + '%';

    if (data.preview && data.preview.length > 0) {
        let table = '<table><thead><tr>';
        const columns = Object.keys(data.preview[0]);
        columns.forEach(col => { table += `<th>${col}</th>`; });
        table += '</tr></thead><tbody>';
        data.preview.forEach(row => {
            table += '<tr>';
            columns.forEach(col => { table += `<td>${row[col]}</td>`; });
            table += '</tr>';
        });
        table += '</tbody></table>';
        document.getElementById('dataPreview').innerHTML = table;
    }
}

function downloadResults() {
    if (!resultsData) { alert('No results to download'); return; }
    const csv = 'Logistics Optimization Results\nCost Savings,' + resultsData.improvements.cost_savings + '%\n';
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'logistics_optimization.csv'; a.click();
}

function downloadReport() {
    alert('PDF report generation will be implemented in your ML notebook!');
}

let uploadedFile = null;
let resultsData = null;

// Setup drag and drop
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');

uploadArea.addEventListener('click', () => {
    fileInput.click();
});

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (!file.name.endsWith('.csv')) {
        alert('Please upload a CSV file');
        return;
    }

    uploadedFile = file;

    // Show file info
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileSize').textContent = formatFileSize(file.size);
    document.getElementById('fileInfo').classList.remove('hidden');

    // Show config panel
    document.getElementById('configPanel').classList.remove('hidden');

    // Hide results if any
    document.getElementById('resultsSection').classList.add('hidden');
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

async function runForecast() {
    if (!uploadedFile) {
        alert('Please upload a file first');
        return;
    }

    // Show loading
    document.getElementById('loadingSection').classList.remove('hidden');
    document.getElementById('resultsSection').classList.add('hidden');

    // Prepare form data
    const formData = new FormData();
    formData.append('file', uploadedFile);
    formData.append('forecast_horizon', document.getElementById('forecastHorizon').value);
    formData.append('model_type', document.getElementById('modelType').value);
    formData.append('confidence_level', document.getElementById('confidenceLevel').value);

    try {
        const response = await fetch('/api/forecast', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            alert('Error: ' + data.error);
            document.getElementById('loadingSection').classList.add('hidden');
            return;
        }

        // Store results
        resultsData = data;

        // Display results
        displayResults(data);

        // Hide loading, show results
        document.getElementById('loadingSection').classList.add('hidden');
        document.getElementById('resultsSection').classList.remove('hidden');

        // Scroll to results
        document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });

    } catch (error) {
        alert('Error: ' + error.message);
        document.getElementById('loadingSection').classList.add('hidden');
    }
}

function displayResults(data) {
    // Display metrics
    document.getElementById('mape').textContent = data.metrics.mape.toFixed(1) + '%';
    document.getElementById('r2').textContent = data.metrics.r2.toFixed(2);
    document.getElementById('rmse').textContent = data.metrics.rmse.toFixed(1);
    document.getElementById('horizonDisplay').textContent = data.forecast_horizon + ' days';
    document.getElementById('modelDisplay').textContent = data.model_type.toUpperCase();

    // Display data preview table
    if (data.preview && data.preview.length > 0) {
        let table = '<table><thead><tr>';

        // Headers
        const columns = Object.keys(data.preview[0]);
        columns.forEach(col => {
            table += `<th>${col}</th>`;
        });
        table += '</tr></thead><tbody>';

        // Rows
        data.preview.forEach(row => {
            table += '<tr>';
            columns.forEach(col => {
                table += `<td>${row[col]}</td>`;
            });
            table += '</tr>';
        });

        table += '</tbody></table>';
        document.getElementById('dataPreview').innerHTML = table;
    }
}

function downloadResults() {
    if (!resultsData) {
        alert('No results to download');
        return;
    }

    // Create CSV content
    let csv = 'Demand Forecast Results\n\n';
    csv += 'Model Type,' + resultsData.model_type + '\n';
    csv += 'Forecast Horizon,' + resultsData.forecast_horizon + ' days\n';
    csv += 'MAPE,' + resultsData.metrics.mape + '%\n';
    csv += 'R² Score,' + resultsData.metrics.r2 + '\n';
    csv += 'RMSE,' + resultsData.metrics.rmse + '\n\n';

    // Download
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'demand_forecast_results.csv';
    a.click();
}

function downloadReport() {
    alert('PDF report generation will be implemented in your ML notebook!\nFor now, download the CSV results.');
}

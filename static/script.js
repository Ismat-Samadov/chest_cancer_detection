// static/script.js
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('upload-form');
    const loading = document.getElementById('loading');
    const resultContainer = document.getElementById('result-container');
    const uploadedImage = document.getElementById('uploaded-image');
    const diagnosisHeader = document.getElementById('diagnosis-header');
    const cancerProbabilityBar = document.getElementById('cancer-probability-bar');
    const cancerProbabilityText = document.getElementById('cancer-probability-text');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceText = document.getElementById('confidence-text');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const fileInput = document.getElementById('file');
        const file = fileInput.files[0];
        
        if (!file) {
            alert('Please select a file to upload');
            return;
        }
        
        // Show loading and hide results
        loading.classList.remove('d-none');
        resultContainer.classList.add('d-none');
        
        // Create form data for upload
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            // Send request to backend
            const response = await fetch('/predict/', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Error processing the image');
            }
            
            const result = await response.json();
            
            // Display results
            displayResults(result, file);
            
        } catch (error) {
            alert('Error: ' + error.message);
            console.error('Error:', error);
        } finally {
            loading.classList.add('d-none');
        }
    });
    
    function displayResults(result, file) {
        // Set image
        uploadedImage.src = URL.createObjectURL(file);
        
        // Set diagnosis
        diagnosisHeader.textContent = `Diagnosis: ${result.diagnosis}`;
        diagnosisHeader.classList.remove('cancer', 'normal');
        diagnosisHeader.classList.add(result.diagnosis.toLowerCase());
        
        // Set cancer probability
        const cancerProbability = result.cancer_probability * 100;
        cancerProbabilityBar.style.width = `${cancerProbability}%`;
        cancerProbabilityBar.classList.remove('bg-success', 'bg-warning', 'bg-danger');
        
        if (cancerProbability < 30) {
            cancerProbabilityBar.classList.add('bg-success');
        } else if (cancerProbability < 70) {
            cancerProbabilityBar.classList.add('bg-warning');
        } else {
            cancerProbabilityBar.classList.add('bg-danger');
        }
        
        cancerProbabilityText.textContent = `${cancerProbability.toFixed(2)}%`;
        
        // Set confidence
        const confidence = result.confidence * 100;
        confidenceBar.style.width = `${confidence}%`;
        confidenceText.textContent = `${confidence.toFixed(2)}%`;
        
        // Show result container
        resultContainer.classList.remove('d-none');
    }
});
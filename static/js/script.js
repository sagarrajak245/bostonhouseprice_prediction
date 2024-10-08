// static/script.js
document.getElementById('predictionForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const inputData = document.getElementById('inputData').value;

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ data: inputData }),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').innerText = 'Predicted Price: ' + data.prediction;
    })
    .catch(error => console.error('Error:', error));
});
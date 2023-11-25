let jsonData; // Variable to store the parsed JSON data

function handleFile() {
    const fileInput = document.getElementById('csvFileInput');

    const file = fileInput.files[0];

    if (file) {
        // Use PapaParse to parse the CSV file
        Papa.parse(file, {
            header: true,
            dynamicTyping: true,
            complete: function(results) {
                // results.data contains the parsed CSV data as an array of objects
                jsonData = results.data;

                // Now you can use jsonData as a JavaScript variable
                console.log('Parsed JSON data:', jsonData);
            },
            error: function(error) {
                console.error('CSV parsing error:', error.message);
            }
        });
    } else {
        alert('Please select a CSV file.');
    }
}
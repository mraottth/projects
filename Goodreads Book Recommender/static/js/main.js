let myRecs;
let myPopular;
let myTopRated;
let view;

document.getElementById('uploadForm').addEventListener('submit', function (e) {
    e.preventDefault();
    
    function removeFadeOut( el, speed ) {
        var seconds = speed/1000;
        el.style.transition = "opacity "+seconds+"s ease";
        el.style.opacity = 0;
        setTimeout(function() {
            el.parentNode.removeChild(el);
        }, speed);
    }

    // Hide upload form
    removeFadeOut(document.getElementById('uploadForm'), 500);                

    // Show the loader
    function showLoader() {
        setTimeout(() => { document.getElementById('loader').style.display = 'block';  }, 510)
    }
    showLoader()

    var formData = new FormData(this);

    fetch('/get_recommendations', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {

        myRecs = data[0]
        myPopular = data[1]
        myTopRated = data[2]

        // Hide the loader after displaying the results
        document.getElementById('loader').style.display = 'none';                    

        // Hide instructions after displaying the results
        document.getElementById('instructions').style.display = 'none';                    

        // Show filters
        document.getElementById('filters').style.display = 'block';
        
        updateTable(myRecs)
    })
    .catch(error => console.error('Error:', error));
});

// Button click listeners
document.getElementById('recs').addEventListener('click', function () {
    view = "recs"
    updateTable(myRecs)
});

document.getElementById('popular').addEventListener('click', function () {
    view = "pop"
    updateTable(myPopular)
});

document.getElementById('topRated').addEventListener('click', function () {
    view = "tr"
    updateTable(myTopRated)
});

function updateTable (dataArray) {
    
    // Display the result in a dynamic table
    var tableHtml = '<table border="1">';

    // Check if there is data
    if (dataArray.data.length > 0) {
        
        let color; 
        if (view == "tr") {
            color = "#0096AB"
        } else if (view == "pop") {
            color = "#00BE9F"
        } else {
            color = "#294170"
        }

        // Use the ordered keys as table headers
        tableHtml += '<tr>';
        dataArray.columns.forEach(key => {
            // replace snake case with title case
            var key_title = key.replace (/^[-_]*(.)/, (_, c) => c.toUpperCase())      
                .replace (/[-_]+(.)/g, (_, c) => ' ' + c.toUpperCase()) 
            
                if (key_title != "Url") {
                    tableHtml += '<th style="background-color:'+ color + '">' + key_title + '</th>';
                }
        });
        tableHtml += '</tr>';

        // Iterate through data and create rows
        dataArray.data.forEach(row => {
            tableHtml += '<tr>';
            // Use the ordered keys to create the columns
            dataArray.columns.forEach(key => {
                
                if (key === 'title') {
                    // Make the value a clickable hyperlink
                    tableHtml += '<td><a href="' + row['url'] + '" target="_blank">' + row[key] + '</a></td>';
                } else if (key === 'url') {
                    // skip column
        
                } else {
                    // Normal cell without a hyperlink
                    tableHtml += '<td>' + row[key] + '</td>';
                }
            });
            tableHtml += '</tr>';
        });
    } else {
        tableHtml += '<tr><td colspan="999">No data available</td></tr>';
    }

    tableHtml += '</table>';

    document.getElementById('resultTable').innerHTML = tableHtml;

    // Show table
    document.getElementById('resultTable').style.display = 'block'; 
 
}
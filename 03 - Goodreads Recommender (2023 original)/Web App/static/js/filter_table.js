// JavaScript for table filtering
document.getElementById('tableFilter').addEventListener('input', function () {
    var filterValue = this.value.toLowerCase();
    var rows = document.getElementById('resultTable').getElementsByTagName('tbody')[0].getElementsByTagName('tr');

    for (var i = 0; i < rows.length; i++) {
        // Skip the first row (header row)
        if (i === 0) {
            continue;
        }

        var rowText = rows[i].textContent.toLowerCase();

        // Check if the row contains the filter value
        if (rowText.includes(filterValue)) {
            rows[i].style.display = '';
        } else {
            rows[i].style.display = 'none';
        }
    }
});
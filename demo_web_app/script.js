// Close Alert Banner
function closeAlert() {
    // Bootstrap handles alert dismissal via data attributes
}

// Open Settings Modal
function openSettings() {
    var myModal = new bootstrap.Modal(document.getElementById('settings-modal'));
    myModal.show();
}

// Close Settings Modal
function closeSettings() {
    var myModalEl = document.getElementById('settings-modal');
    var modal = bootstrap.Modal.getInstance(myModalEl);
    modal.hide();
}

// Update Threshold Label
function updateThresholdLabel(value) {
    document.getElementById('threshold-label').innerText = value;
}

// Save Settings
function saveSettings() {
    // Save settings logic here
    alert('Settings saved!');
    closeSettings();
}

// Open Support
function openSupport() {
    window.location.href = 'mailto:support@aquaalert.com';
}

// Initialize Charts and Maps when DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Prediction Chart
    var ctxPrediction = document.getElementById('prediction-chart').getContext('2d');
    var predictionChart = new Chart(ctxPrediction, {
        type: 'line',
        data: {
            labels: ['Now', '+6h', '+12h', '+18h', '+24h'],
            datasets: [{
                label: 'Algae Level Prediction',
                data: [70, 75, 80, 85, 90],
                backgroundColor: 'rgba(255, 193, 7, 0.2)',
                borderColor: 'rgba(255, 193, 7, 1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true,
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false,
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });

    // Trend Chart
    var ctxTrend = document.getElementById('trend-chart').getContext('2d');
    var trendChart = new Chart(ctxTrend, {
        type: 'bar',
        data: {
            labels: ['Day -4', 'Day -3', 'Day -2', 'Yesterday', 'Today'],
            datasets: [{
                label: 'Algae Level',
                data: [50, 55, 60, 65, 70],
                backgroundColor: 'rgba(220, 53, 69, 0.8)',
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false,
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });

    // Growth Rate Chart
    var ctxGrowth = document.getElementById('growth-chart').getContext('2d');
    var growthChart = new Chart(ctxGrowth, {
        type: 'line',
        data: {
            labels: ['Day -4', 'Day -3', 'Day -2', 'Yesterday', 'Today'],
            datasets: [{
                label: 'Algae Growth Rate',
                data: [10, 20, 35, 55, 80],
                backgroundColor: 'rgba(0, 168, 107, 0.2)',
                borderColor: 'rgba(0, 168, 107, 1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true,
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false,
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });

    // Initialize Main Map using Leaflet.js
    var map = L.map('map').setView([68.0, 16.0], 6); // Centered in the Norwegian Sea

    // Add OpenStreetMap tiles
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);

    // Algae Icon
    var algaeIcon = L.icon({
        iconUrl: 'images/algae-icon.png',
        iconSize: [30, 30],
        iconAnchor: [15, 30],
    });

    // Add Algae Marker (Starting Point in the Sea)
    var algaeStartPoint = [68.0, 12.0]; // Starting point in the sea
    L.marker(algaeStartPoint, { icon: algaeIcon }).addTo(map)
        .bindPopup('<strong>Algae Bloom Detected</strong><br>Monitoring initiated.');

    // Fish Farm Icon
    var fishFarmIcon = L.icon({
        iconUrl: 'images/fish-farm-icon.png', // You need to provide this icon
        iconSize: [30, 30],
        iconAnchor: [15, 30],
    });

    // Add Fish Farm Location near Tromsø
    var fishFarmLocation = [69.6, 18.8]; // Near Tromsø
    L.marker(fishFarmLocation, { icon: fishFarmIcon }).addTo(map)
        .bindPopup('<strong>Your Fish Farm</strong>');

    // Add Predicted Movement to the Main Map
    var algaeCoordinates = [
        [68.0, 12.0], // Day 1 - Starting point
        [68.5, 13.5], // Day 2
        [69.0, 15.0], // Day 3
        [69.3, 16.9], // Day 4 - Close to Tromsø but not hitting it
    ];

    L.polyline(algaeCoordinates, { color: 'red', dashArray: '5, 10' }).addTo(map);

    algaeCoordinates.forEach(function(coord, index) {
        L.circleMarker(coord, {
            radius: 4,
            fillColor: 'blue',
            color: 'white',
            weight: 1,
            opacity: 1,
            fillOpacity: 0.8
        }).addTo(map)
        .bindPopup('Day ' + (index + 1) + ' Prediction');
    });

    // Initialize Predicted Movement Map
    var movementMap = L.map('movement-map').setView([68.0, 16.0], 6);

    // Add OpenStreetMap tiles to movement map
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors'
    }).addTo(movementMap);

    // Add a Polyline to show predicted movement
    var movementLine = L.polyline(algaeCoordinates, { color: 'red' }).addTo(movementMap);

    // Add markers for each predicted position
    algaeCoordinates.forEach(function(coord, index) {
        L.circleMarker(coord, {
            radius: 5,
            fillColor: index === algaeCoordinates.length - 1 ? 'red' : 'orange',
            color: 'white',
            weight: 1,
            opacity: 1,
            fillOpacity: 0.8
        }).addTo(movementMap)
        .bindPopup('Day ' + (index + 1));
    });

    // Add Fish Farm Location to Predicted Movement Map
    L.marker(fishFarmLocation, { icon: fishFarmIcon }).addTo(movementMap)
        .bindPopup('<strong>Your Fish Farm</strong>');

    // Fit map to show both the algae path and the fish farm
    var allCoordinates = algaeCoordinates.concat([fishFarmLocation]);
    var bounds = L.latLngBounds(allCoordinates);
    movementMap.fitBounds(bounds);
});

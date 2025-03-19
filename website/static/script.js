let ctx = document.getElementById("alertChart").getContext("2d");
let alertChart = new Chart(ctx, {
    type: "line",
    data: {
        labels: [],
        datasets: [
            {
                label: "Actual Alert_11",
                data: [],
                borderColor: "blue",
                borderWidth: 2,
                fill: false
            },
            {
                label: "Predicted Alert_11",
                data: [],
                borderColor: "red",
                borderWidth: 2,
                fill: false
            }
        ]
    },
    options: {
        responsive: true,
        scales: {
            x: { title: { display: true, text: "Time" } },
            y: { title: { display: true, text: "Alert_11 Value" } }
        }
    }
});

function updateChart() {
    fetch("/predict")
        .then(response => response.json())
        .then(data => {
            if (data.error) return;

            let labels = data.map((_, i) => i + 1);
            let actualData = data.map(d => d.actual);
            let predictedData = data.map(d => d.predicted);

            alertChart.data.labels = labels;
            alertChart.data.datasets[0].data = actualData;
            alertChart.data.datasets[1].data = predictedData;
            alertChart.update();
        })
        .catch(error => console.error("Error fetching data:", error));
}

setInterval(updateChart, 10000);

// const ctx = document.getElementById('trainingChart').getContext('2d');

// // Fetch the JSON file
// fetch('history.json')
//     .then(response => response.json())
//     .then(history => {
//         // Create the chart
//         const trainingChart = new Chart(ctx, {
//             type: 'line',
//             data: {
//                 labels: Array.from({ length: history.loss.length }, (_, i) => i + 1), // Epochs
//                 datasets: [
//                     {
//                         label: 'Training Loss',
//                         data: history.loss,
//                         borderColor: 'rgba(255, 99, 132, 1)',
//                         backgroundColor: 'rgba(255, 99, 132, 0.2)',
//                         fill: false,
//                         lineTension: 0.1
//                     },
//                     {
//                         label: 'Validation Loss',
//                         data: history.val_loss,
//                         borderColor: 'rgba(54, 162, 235, 1)',
//                         backgroundColor: 'rgba(54, 162, 235, 0.2)',
//                         fill: false,
//                         lineTension: 0.1
//                     },
//                     {
//                         label: 'Training Accuracy',
//                         data: history.accuracy,
//                         borderColor: 'rgba(75, 192, 192, 1)',
//                         backgroundColor: 'rgba(75, 192, 192, 0.2)',
//                         fill: false,
//                         lineTension: 0.1
//                     },
//                     {
//                         label: 'Validation Accuracy',
//                         data: history.val_accuracy,
//                         borderColor: 'rgba(153, 102, 255, 1)',
//                         backgroundColor: 'rgba(153, 102, 255, 0.2)',
//                         fill: false,
//                         lineTension: 0.1
//                     }
//                 ]
//             },
//             options: {
//                 responsive: true,
//                 scales: {
//                     x: {
//                         title: {
//                             display: true,
//                             text: 'Epoch'
//                         }
//                     },
//                     y: {
//                         title: {
//                             display: true,
//                             text: 'Value'
//                         }
//                     }
//                 }
//             }
//         });
//     })
//     .catch(error => console.error('Error loading the JSON file:', error));

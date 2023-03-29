// if (!window.dash_clientside) {
//     window.dash_clientside = {};
// }
// window.dash_clientside.updateColors = function (_, bgCol, pltBgCol, txtCol, plotIds) {
//     // Update colors of main dashboard app
//     let dashboard = document.getElementById("app");
//     if (dashboard) {
//         dashboard.style.backgroundColor = bgCol.hex;
//         dashboard.style.color = txtCol.hex;
//     }
//     // Update plot colors
//     let parsedPlotIds = JSON.parse(plotIds);
//     for (let i = 0; i < parsedPlotIds.length; i++) {
//         let plot = document.getElementById(parsedPlotIds[i]);
//         if (plot) {
//             Plotly.relayout(plot, {
//                 'paper_bgcolor': bgCol.hex,
//                 'plot_bgcolor': pltBgCol.hex,
//                 'font.color': txtCol.hex,
//                 'xaxis.titlefont.color': txtCol.hex,
//                 'yaxis.titlefont.color': txtCol.hex,
//                 'xaxis.tickfont.color': txtCol.hex,
//                 'yaxis.tickfont.color': txtCol.hex
//             });
//         }
//     }
// };

// function updateColors(bgCol, pltBgCol, txtCol, plotIds) {
//     // Update colors of main dashboard app
//     let dashboard = document.getElementById("app");
//     if (dashboard) {
//         dashboard.style.backgroundColor = bgCol;
//         dashboard.style.color = txtCol;
//     }
//     // Update plot colors
//     let parsedPlotIds = JSON.parse(plotIds);
//     for (let i = 0; i < parsedPlotIds.length; i++) {
//         let plot = document.getElementById(parsedPlotIds[i]);
//         if (plot) {
//             Plotly.relayout(plot, {
//                 'paper_bgcolor': bgCol,
//                 'plot_bgcolor': pltBgCol,
//                 'font.color': txtCol,
//                 'xaxis.titlefont.color': txtCol,
//                 'yaxis.titlefont.color': txtCol,
//                 'xaxis.tickfont.color': txtCol,
//                 'yaxis.tickfont.color': txtCol
//             });
//         }
//     }
// }

// document.addEventListener("DOMContentLoaded", function () {
//     let button = document.getElementById("col-button");
//     if (button) {
//         button.addEventListener("click", function () {
//             let bgColPicker = document.getElementById("dash-bg-col-picker");
//             let txtColPicker = document.getElementById("dash-txt-col-picker");
//             let pltBgColPicker = document.getElementById("plt-bg-col-picker");
//             let plotIdsStorage = document.getElementById("plot-ids-storage");

//             let bgCol = bgColPicker ? JSON.parse(bgColPicker.value).hex : "#050505";
//             let txtCol = txtColPicker ? JSON.parse(txtColPicker.value).hex : "#f2f2f2";
//             let pltBgCol = pltBgColPicker ? JSON.parse(pltBgColPicker.value).hex : "#0d0d0d";
//             let plotIds = plotIdsStorage ? plotIdsStorage.textContent : "[]";

//             updateColors(bgCol, pltBgCol, txtCol, plotIds);
//         });
//     }
// });

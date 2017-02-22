$(function () {

    var emb_json = JSON.parse(emb);
    $('#container').highcharts({
        chart: {
            type: 'scatter',
            zoomType: 'xy',
            height: 900,
            width: 1200
        },
        title: {
            text: 'Trained Diagnosis Vector'
        },
        subtitle: {
            text: 'Dimension reduced by TSNE'
        },
        xAxis: {
            title: {
                enabled: true,
                text: 'TSNE Dimension 1'
            },
            startOnTick: true,
            endOnTick: true,
            showLastLabel: true
        },
        yAxis: {
            title: {
                text: 'TSNE Dimension 2'
            }
        },
        legend: {
            enabled: false
        },
        plotOptions: {
            scatter: {
                marker: {
                    radius: 5,
                    states: {
                        hover: {
                            enabled: true,
                            lineColor: 'rgb(100,100,100)'
                        }
                    }
                },
                states: {
                    hover: {
                        marker: {
                            enabled: false
                        }
                    }
                },
                tooltip: {
                    headerFormat: '<b>{series.name}</b><br>',
                    pointFormat: '{point.x}, {point.y}'
                }
            }
        },
        series: emb_json
    });
});

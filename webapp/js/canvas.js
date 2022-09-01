var ctx, signaturePad;

function Init() {
    var canvas = document.getElementById('myCanvas');
    signaturePad = new SignaturePad(canvas, {
        minWidth: 20,
        maxWidth: 25
    });
    ctx = canvas.getContext('2d');
};

function sendClassifyRequest() {
    var data = ctx.getImageData(0, 0, 500, 500).data;
    var opacities = data.filter(function(element, idx) {
        return (idx % 4 === 3);
    });

    // opacities is a Uint8ClampedArray, which gets weirdly JSON serialized
    var opacitiesToSend = Array.from(opacities)

    $.ajax({
        url: '/classify',
        type: 'POST',
        data: JSON.stringify(
             {
                 imageData: opacitiesToSend,
                 returnAllProbabilities: true,
                 computeGradients: true
             }
        ),
        contentType:"application/json; charset=utf-8",
        dataType: 'json',
        processData: false,
        success: function(response) {
            $('#classificationResult').text(response.classification);
            $('#classProbabilities').text(response.probabilities.map((x) => x.toFixed(3)));
            var gradientData = [{
              z: response.gradients[response.classification],
              type: 'heatmap'
            }];
            Plotly.newPlot(
                'gradientHeatmap',
                gradientData,
                {
                  title: 'Gradients of p(' + response.classification + ')',
                  width: 700,
                  height: 700,
                  yaxis: {
                    autorange: 'reversed',
                  }
                }
            );
        }
    });

}

function clearAll() {
    signaturePad.clear();
    $('#classificationResult').text('');
    $('#classProbabilities').text('');
    $('#gradientHeatmap').empty();
}
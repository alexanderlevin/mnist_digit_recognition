var ctx;

function Init() {
    var canvas = document.getElementById('myCanvas');
    var signaturePad = new SignaturePad(canvas, {
        minWidgth: 20,
        maxWidth: 25
    });
    ctx = canvas.getContext('2d');
};

function sendClassifyRequest() {
    var data = ctx.getImageData(0, 0, 500, 500).data;
    var opacities = data.filter(function(element, idx) {
        return (idx % 4 === 3);
    });

    var opacitiesToSend = []
    opacities.forEach(function(e) {
        opacitiesToSend.push(e);
    });

    $.ajax({
        url: '/classify',
        type: 'POST',
        data: JSON.stringify(opacitiesToSend),
        contentType:"application/json; charset=utf-8",
        dataType: 'json',
        processData: false,
        success: function(response) {
            $('#classificationResult').text(
                response.classification);
        }
    });

}
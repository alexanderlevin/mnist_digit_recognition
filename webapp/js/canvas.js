var mousePressed = false,
    lastX, lastY,
    ctx;

function InitThis() {
	console.log("InitThis");
    ctx = document.getElementById('myCanvas').getContext('2d');
    ctx.lineWidth = 30;
    $('#myCanvas').mousedown(function(e) {
        mousePressed = true;
        Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, false);
    });
    $('#myCanvas').mousemove(function(e) {
        if(mousePressed) {
            Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, true);
        }
    });

	$('#myCanvas').mouseup(function(e) {
        mousePressed = false;
    });
    $('#myCanvas').mouseleave(function(e) {
        mousePressed = false;
    });
};

function Draw(x, y, isDown) {
	if(isDown) {
		ctx.beginPath();
		ctx.moveTo(lastX, lastY);
		ctx.lineTo(x, y);
		ctx.closePath();
		ctx.stroke()


	}
	lastX = x;
	lastY = y;

}

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
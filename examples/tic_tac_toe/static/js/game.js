var player1='x';

$(document).ready(function(){
    $('#game td').click(function(){
        var clickedBtnID = $(this).attr('id');
        var p = $(this).text();
        if (p == " ") {
            $(this).html(player1);
            var tboard = "";
            $('#game td').each(function( index ) {
                tboard += $( this ).text()  + ","
            });
            tboard = tboard.substring(0, tboard.length - 1);
            $.ajax({
                type: 'GET',
                url: 'play',
                data: { board: tboard },
                dataType: 'json',
                success: function (data) {
                    $("#b1").html(data["b1"])
                    $("#b2").html(data["b2"])
                    $("#b3").html(data["b3"])
                    $("#b4").html(data["b4"])
                    $("#b5").html(data["b5"])
                    $("#b6").html(data["b6"])
                    $("#b7").html(data["b7"])
                    $("#b8").html(data["b8"])
                    $("#b9").html(data["b9"])
                    if (data["game over"] == "true") {
                        $('#game td').unbind( "click" );
                        $('.game_over').removeClass('game_over');
                    }
                }
            });
        }
    });
});

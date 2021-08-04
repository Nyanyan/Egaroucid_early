function disableScroll(event) {
    event.preventDefault();
}
document.addEventListener('touchmove', disableScroll, {passive: false});
function move(r, c) {
    alert(r + " " + c);
    $.ajax({
        type: "POST",
        url: "/move",
        data: {r: r, c: c},
        async: false,
        dataType: "json",
    }).done(function(data) {
        const grid = JSON.parse(data.values);
        var table = document.getElementById("board");
        for (var y = 0; y < 8; y++) {
            for (var x = 0; x < 8; x++) {
                alert(grid[y * 8 + x]);
            }
        }
        console.log("done");
        alert("done");
    }).fail(function(data) {
        console.log("fail");
        alert("An error occurred. Please try again.")
    });
}
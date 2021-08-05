function disableScroll(event) {
    event.preventDefault();
}
document.addEventListener('touchmove', disableScroll, {passive: false});
function move(r, c) {
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
                if (grid[y * 8 + x] == 0) {
                    table.rows[y].cells[x].innerHTML = '<span class="black_stone"></span>';
                    table.rows[y].cells[x].setAttribute('onclick', "");
                } else if (grid[y * 8 + x] == 1) {
                    table.rows[y].cells[x].innerHTML = '<span class="white_stone"></span>';
                    table.rows[y].cells[x].setAttribute('onclick', "");
                } else if (grid[y * 8 + x] == 2) {
                    table.rows[y].cells[x].innerHTML = '<span class="legal_stone"></span>';
                    table.rows[y].cells[x].setAttribute('onclick', "move(this.parentNode.rowIndex, this.cellIndex)");
                } else {
                    table.rows[y].cells[x].innerHTML = '<span class="empty_stone"></span>';
                    table.rows[y].cells[x].setAttribute('onclick', "");
                }
            }
        }
        table = document.getElementById("status");
        table.rows[0].cells[2].firstChild.innerHTML = grid[64];
        table.rows[0].cells[4].firstChild.innerHTML = grid[65];
        if (grid[66] == 0) {
            table.rows[0].cells[0].innerHTML = '<span class="legal_stone"></span>';
            table.rows[0].cells[6].innerHTML = '<span class="state_blank"></span>';
        } else if (grid[66] == 1) {
            table.rows[0].cells[0].innerHTML = '<span class="state_blank"></span>';
            table.rows[0].cells[6].innerHTML = '<span class="legal_stone"></span>';
        } else {
            table.rows[0].cells[0].innerHTML = '<span class="state_blank"></span>';
            table.rows[0].cells[6].innerHTML = '<span class="state_blank"></span>';
        }
        console.log("done");
    }).fail(function(data) {
        console.log("fail");
        alert("An error occurred. Please try again.")
    });
}
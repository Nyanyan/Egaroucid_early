/* global BigInt */

function disableScroll(event) {
    event.preventDefault();
}

document.addEventListener('touchmove', disableScroll, {passive: false});

var hw = 8;
let dy = [0, 1, 0, -1, 1, 1, -1, -1];
let dx = [1, 0, -1, 0, 1, -1, 1, -1];
let grid = [
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1]
];
var player = 0;

function show() {
    var table = document.getElementById("board");
    if (!check_mobility()) {
        player = 1 - player;
        if (!check_mobility()) {
            player = 2;
        }
    }
    for (var y = 0; y < 8; ++y) {
        for (var x = 0; x < 8; ++x) {
            if (grid[y][x] == 0) {
                table.rows[y].cells[x].innerHTML = '<span class="black_stone"></span>';
                table.rows[y].cells[x].setAttribute('onclick', "");
            } else if (grid[y][x] == 1) {
                table.rows[y].cells[x].innerHTML = '<span class="white_stone"></span>';
                table.rows[y].cells[x].setAttribute('onclick', "");
            } else if (grid[y][x] == 2) {
                table.rows[y].cells[x].innerHTML = '<span class="legal_stone"></span>';
                table.rows[y].cells[x].setAttribute('onclick', "move(this.parentNode.rowIndex, this.cellIndex)");
            } else {
                table.rows[y].cells[x].innerHTML = '<span class="empty_stone"></span>';
                table.rows[y].cells[x].setAttribute('onclick', "");
            }
        }
    }
    var black_count = 0, white_count = 0;
    for (var y = 0; y < hw; ++y) {
        for (var x = 0; x < hw; ++x) {
            if (grid[y][x] == 0)
                ++black_count;
            else if (grid[y][x] == 1)
                ++white_count;
        }
    }
    table = document.getElementById("status");
    table.rows[0].cells[2].firstChild.innerHTML = black_count;
    table.rows[0].cells[4].firstChild.innerHTML = white_count;
    if (player == 0) {
        table.rows[0].cells[0].innerHTML = '<span class="legal_stone"></span>';
        table.rows[0].cells[6].innerHTML = '<span class="state_blank"></span>';
    } else if (player == 1) {
        table.rows[0].cells[0].innerHTML = '<span class="state_blank"></span>';
        table.rows[0].cells[6].innerHTML = '<span class="legal_stone"></span>';
    } else {
        table.rows[0].cells[0].innerHTML = '<span class="state_blank"></span>';
        table.rows[0].cells[6].innerHTML = '<span class="state_blank"></span>';
    }
}

function empty(y, x) {
    return grid[y][x] == -1 || grid[y][x] == 2;
}

function inside(y, x) {
    return 0 <= y && y < hw && 0 <= x && x < hw;
}

function check_mobility() {
    var res = false;
    for (var y = 0; y < hw; ++y) {
        for (var x = 0; x < hw; ++x) {
            if (!empty(y, x))
                continue;
            grid[y][x] = -1;
            for (var dr = 0; dr < 8; ++dr) {
                var ny = y + dy[dr];
                var nx = x + dx[dr];
                if (!inside(ny, nx))
                    continue;
                if (empty(ny, nx))
                    continue;
                if (grid[ny][nx] == player)
                    continue;
                var flag = false;
                var nny = ny, nnx = nx;
                var plus = 0;
                for (var d = 0; d < hw; ++d) {
                    if (!inside(nny, nnx))
                        break;
                    if (empty(nny, nnx))
                        break;
                    if (grid[nny][nnx] == player) {
                        flag = true;
                        break;
                    }
                    nny += dy[dr];
                    nnx += dx[dr];
                    ++plus;
                }
                if (flag) {
                    grid[y][x] = 2;
                    res = true;
                    break;
                }
            }
        }
    }
    return res;
}

window.onload = function init() {
    grid[3][3] = 1
    grid[3][4] = 0
    grid[4][3] = 0
    grid[4][4] = 1
    player = 0;
    var table = document.getElementById('board');
    for (var y = 0; y < hw; ++y) {
        var row = document.createElement('tr');
        for (var x = 0; x < hw; ++x) {
            var cell = document.createElement('td');
            cell.className = "cell";
            var stone = document.createElement('span');
            stone.className = "empty_stone";
            cell.appendChild(stone);
            row.appendChild(cell);
        }
        table.appendChild(row);
    }
    show();
}

function move(y, x) {
    grid[y][x] = player;
    for (var dr = 0; dr < 8; ++dr) {
        var ny = y + dy[dr];
        var nx = x + dx[dr];
        if (!inside(ny, nx))
            continue;
        if (empty(ny, nx))
            continue;
        if (grid[ny][nx] == player)
            continue;
        var flag = false;
        var nny = ny, nnx = nx;
        var plus = 0;
        for (var d = 0; d < hw; ++d) {
            if (!inside(nny, nnx))
                break;
            if (empty(nny, nnx))
                break;
            if (grid[nny][nnx] == player) {
                flag = true;
                break;
            }
            nny += dy[dr];
            nnx += dx[dr];
            ++plus;
        }
        if (flag) {
            for (var d = 0; d < plus; ++d) {
                grid[ny + d * dy[dr]][nx + d * dx[dr]] = player;
            }
        }
    }
    player = 1 - player;
    show();
}

/*
function move(r, c) {
    $.ajax({
        type: "POST",
        url: "/move",
        data: {r: r, c: c},
        async: false,
        dataType: "json",
    }).done(function(data) {
        const received_data = JSON.parse(data.values);
        var table = document.getElementById("board");
        for (var y = 0; y < 8; y++) {
            for (var x = 0; x < 8; x++) {
                if (received_data[y * 8 + x] == 0) {
                    table.rows[y].cells[x].innerHTML = '<span class="black_stone"></span>';
                    table.rows[y].cells[x].setAttribute('onclick', "");
                } else if (received_data[y * 8 + x] == 1) {
                    table.rows[y].cells[x].innerHTML = '<span class="white_stone"></span>';
                    table.rows[y].cells[x].setAttribute('onclick', "");
                } else if (received_data[y * 8 + x] == 2) {
                    table.rows[y].cells[x].innerHTML = '<span class="legal_stone"></span>';
                    table.rows[y].cells[x].setAttribute('onclick', "move(this.parentNode.rowIndex, this.cellIndex)");
                } else {
                    table.rows[y].cells[x].innerHTML = '<span class="empty_stone"></span>';
                    table.rows[y].cells[x].setAttribute('onclick', "");
                }
            }
        }
        table = document.getElementById("status");
        table.rows[0].cells[2].firstChild.innerHTML = received_data[64];
        table.rows[0].cells[4].firstChild.innerHTML = received_data[65];
        if (received_data[66] == 0) {
            table.rows[0].cells[0].innerHTML = '<span class="legal_stone"></span>';
            table.rows[0].cells[6].innerHTML = '<span class="state_blank"></span>';
        } else if (received_data[66] == 1) {
            table.rows[0].cells[0].innerHTML = '<span class="state_blank"></span>';
            table.rows[0].cells[6].innerHTML = '<span class="legal_stone"></span>';
        } else {
            table.rows[0].cells[0].innerHTML = '<span class="state_blank"></span>';
            table.rows[0].cells[6].innerHTML = '<span class="state_blank"></span>';
        }
        document.getElementById("turn").innerHTML = received_data[67];
        console.log("done");
    }).fail(function(data) {
        console.log("fail");
        alert("An error occurred. Please try again.")
    });
}
*/
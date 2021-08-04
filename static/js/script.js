function disableScroll(event) {
    event.preventDefault();
}
document.addEventListener('touchmove', disableScroll, {passive: false});
function move(r, c) {
    var post = $.ajax({
        type: "POST",
        url: "/move",
        data: {r: r, c: c},
        async: false,
        dataType: "json",
        success: function(response) {
            console.log(response);
            count = response.count;
        }, 
        error: function(error) {
            console.log("Error occurred in move().");
            console.log(error);
            alert("An error occurred. Please try again.");
        }
    })
}
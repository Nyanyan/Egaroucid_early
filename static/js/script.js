function disableScroll(event) {
    event.preventDefault();
}
document.addEventListener('touchmove', disableScroll, { passive: false });
AOS.init();

function toggleSearch() {
    const searchBox = document.getElementById('searchBox');
    if (searchBox.style.display === 'block') {
        searchBox.style.display = 'none';
    } else {
        searchBox.style.display = 'block';
    }
}

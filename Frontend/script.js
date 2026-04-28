document.addEventListener('DOMContentLoaded', () => {
    const tabBrowse = document.getElementById('tab-browse');
    const tabPaste = document.getElementById('tab-paste');
    const sectionBrowse = document.getElementById('section-browse');
    const sectionPaste = document.getElementById('section-paste');

    tabBrowse.addEventListener('click', () => {
        tabBrowse.classList.add('active');
        tabPaste.classList.remove('active');
        sectionBrowse.style.display = 'block';
        sectionPaste.style.display = 'none';
    });

    tabPaste.addEventListener('click', () => {
        tabPaste.classList.add('active');
        tabBrowse.classList.remove('active');
        sectionPaste.style.display = 'block';
        sectionBrowse.style.display = 'none';
    });
});

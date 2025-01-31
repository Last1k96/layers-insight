function initializeSplitter(splitter) {
    const previousSibling = splitter.previousElementSibling;
    const nextSibling = splitter.nextElementSibling;

    if (!previousSibling || !nextSibling) {
        console.error(`Splitter element must have a previous and next sibling.`);
        return;
    }

    function setElementStyles(el, styles) {
        for (const [key, value] of Object.entries(styles)) {
            el.style[key] = value;
        }
    }

    setElementStyles(splitter, {
        boxSizing: 'border-box',
    });

    let isDragging = false;
    let startX = 0;
    let startWidthPrev = 0;
    let startWidthNext = 0;

    splitter.addEventListener('mousedown', (e) => {
        isDragging = true;
        startX = e.clientX;
        startWidthPrev = previousSibling.offsetWidth;
        startWidthNext = nextSibling.offsetWidth;
        document.body.style.cursor = 'col-resize';
    });

    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        const deltaX = e.clientX - startX;
        const newWidthPrev = startWidthPrev + deltaX;
        const newWidthNext = startWidthNext - deltaX;
        previousSibling.style.width = `${newWidthPrev}px`;
        nextSibling.style.width = `${newWidthNext}px`;
    });

    document.addEventListener('mouseup', () => {
        isDragging = false;
        document.body.style.cursor = 'auto';
    });
}

const observer = new MutationObserver((mutations, obs) => {
    // Try to get the splitter element by its id
    const splitter = document.getElementById('splitter');
    if (splitter) {
        console.log('Splitter element detected via MutationObserver.');
        // Initialize the splitter now that it exists
        initializeSplitter(splitter);
        // Once initialized, disconnect the observer so it stops watching
        obs.disconnect();
    }
});

// Start observing the document body for changes to its children (and subtrees)
observer.observe(document.body, {
    childList: true,
    subtree: true
});

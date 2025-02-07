(function() {
    // This function will run once both handles are found in the DOM.
    function initializeDragHandles(leftHandle, rightHandle) {
        console.log('Initializing drag handles.');

        const leftPanel = document.getElementById('left-panel');
        const rightPanel = document.getElementById('right-panel');

        let isDraggingLeft = false;
        let isDraggingRight = false;
        let startX = 0;
        let startWidth = 0;

        // LEFT PANEL DRAG
        leftHandle.addEventListener('mousedown', function(e) {
            isDraggingLeft = true;
            startX = e.clientX;
            startWidth = leftPanel.getBoundingClientRect().width;
            document.body.style.userSelect = 'none'; // prevent text selection
        });

        // RIGHT PANEL DRAG
        rightHandle.addEventListener('mousedown', function(e) {
            isDraggingRight = true;
            startX = e.clientX;
            startWidth = rightPanel.getBoundingClientRect().width;
            document.body.style.userSelect = 'none';
        });

        // MOUSEMOVE
        document.addEventListener('mousemove', function(e) {
            if (!isDraggingLeft && !isDraggingRight) return;
            const dx = e.clientX - startX;

            if (isDraggingLeft) {
                // Expand/collapse the left panel to the right
                let newWidth = startWidth + dx;
                if (newWidth < 100) newWidth = 100;  // clamp min
                leftPanel.style.width = newWidth + "px";
            }
            else if (isDraggingRight) {
                // Expand/collapse the right panel to the left
                let newWidth = startWidth - dx;
                if (newWidth < 100) newWidth = 100;  // clamp min
                rightPanel.style.width = newWidth + "px";
            }
        });

        // MOUSEUP
        document.addEventListener('mouseup', function() {
            isDraggingLeft = false;
            isDraggingRight = false;
            document.body.style.userSelect = 'auto';
        });
    }

    // The MutationObserver will watch for our drag handles to appear in the DOM.
    const observer = new MutationObserver((mutations, obs) => {
        // Try to get the drag handles by their IDs
        const leftHandle = document.getElementById('left-drag-handle');
        const rightHandle = document.getElementById('right-drag-handle');

        // If both exist, we can initialize
        if (leftHandle && rightHandle) {
            console.log('Detected left-drag-handle and right-drag-handle via MutationObserver.');
            initializeDragHandles(leftHandle, rightHandle);
            obs.disconnect(); // Stop observing once initialized
        }
    });

    // Start observing the entire document for changes to its children (and subtrees).
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
})();

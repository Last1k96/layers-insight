window.centerOnNode = function(node_id) {
    console.log("Called centerOnNode with:", node_id)
    try {
        const cy = document.getElementById('ir-graph').__cy;
        if (cy) {
            const node = cy.getElementById(node_id);
            if (node.length > 0) {
                cy.animate({
                    center: { eles: node },
                    duration: 500,
                    easing: 'ease-in-out-cubic'
                });
                return true;
            }
        }
    } catch (e) {
        console.error("Error centering on node:", e);
    }
    return false;
};
def parse_openvino_ir(ir_path: str):
    """
    Parse the OpenVINO IR file at `ir_path` and return a list of Cytoscape elements.
    This is just a placeholder returning a static example.
    Replace with your real parsing logic.
    """
    # TODO: load IR, build node/edge data
    return [
        {'data': {'id': 'node1', 'label': 'Node 1'}},
        {'data': {'id': 'node2', 'label': 'Node 2'}},
        {'data': {'id': 'node3', 'label': 'Node 3'}},
        {'data': {'source': 'node1', 'target': 'node2'}},
        {'data': {'source': 'node2', 'target': 'node3'}}
    ]

# my_dash_ir_app/openvino_graph.py

import xml.etree.ElementTree as ET

def parse_openvino_ir(xml_file_path):
    """
    Parse an OpenVINO IR .xml file and return Cytoscape elements (nodes + edges).
    We also parse the shape attribute to show in the node label.
    """

    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    layers_elem = root.find('layers')
    edges_elem = root.find('edges')

    if layers_elem is None or edges_elem is None:
        raise ValueError("Invalid IR structure: missing <layers> or <edges>")

    node_elements = []
    for layer in layers_elem.findall('layer'):
        layer_id = layer.get('id')
        layer_name = layer.get('name')
        layer_type = layer.get('type')

        # Try to find a <data> sub-element to get the shape attribute.
        data_elem = layer.find('data')
        shape_str = None
        if data_elem is not None:
            shape_str = data_elem.get('shape')  # e.g. "48,3,3,3"

        # Build a user-friendly label. Netron often shows the op name + shape or weights
        # For example: "Convolution\nweights: 48×3×3×3"
        # Or for Add: "Add\nB: 1×48×1×1"
        # You can decide how to parse shape or display it; here’s a simple approach:
        if shape_str:
            display_label = f"{layer_type}\n{shape_str}"
        else:
            display_label = layer_type

        # You might also want to show the original layer name, e.g.:
        #   display_label = f"{layer_name}\n({layer_type}: {shape_str})"
        # Adjust as needed.

        node_elements.append({
            'data': {
                'id': layer_id,
                'type': layer_type,         # used for stylesheet-based coloring
                'display_label': display_label
            }
        })

    edge_elements = []
    for edge in edges_elem.findall('edge'):
        from_layer = edge.get('from-layer')
        to_layer = edge.get('to-layer')
        edge_elements.append({
            'data': {
                'source': from_layer,
                'target': to_layer
            }
        })

    return node_elements + edge_elements

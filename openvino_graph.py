# my_dash_ir_app/openvino_graph.py

import xml.etree.ElementTree as ET

def parse_openvino_ir(xml_file_path):
    """
    Parse an OpenVINO IR .xml file and return a list of Cytoscape elements (nodes + edges).
    Each 'layer' is a node; each 'edge' is a directed edge from-layer -> to-layer.
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
        label_str = f"{layer_name}\n({layer_type})"

        node_elements.append({
            'data': {
                'id': layer_id,
                'label': label_str
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

import xml.etree.ElementTree as ET

def parse_openvino_ir(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    layers_elem = root.find('layers')
    edges_elem = root.find('edges')

    if layers_elem is None or edges_elem is None:
        raise ValueError("Invalid IR structure: missing <layers> or <edges>")

    shape_dict = {}

    for layer in layers_elem.findall('layer'):
        layer_id = layer.get('id')

        # Parse output ports
        output_elem = layer.find('output')
        if output_elem is not None:
            for port in output_elem.findall('port'):
                port_id = port.get('id')
                dims = [dim.text for dim in port.findall('dim')]
                shape_str = "x".join(dims) if dims else "?"
                shape_dict[(layer_id, port_id)] = shape_str

    node_elements = []
    ignore_node_id = []
    for layer in layers_elem.findall('layer'):
        layer_id = layer.get('id')
        layer_name = layer.get('name')
        layer_type = layer.get('type')
        if layer_type == 'Const':
            ignore_node_id.append(layer_id)
            continue

        node_elements.append({
            'data': {
                'id': layer_id,
                'type': layer_type,
                'display_label': layer_type,
                'layer_name': layer_name,
                'border_color': "#242424"
            }
        })

    edge_elements = []
    for edge in edges_elem.findall('edge'):
        from_layer = edge.get('from-layer')
        from_port = edge.get('from-port')
        to_layer = edge.get('to-layer')

        if ignore_node_id.count(from_layer) != 0 or ignore_node_id.count(to_layer) != 0:
            continue

        edge_shape_str = shape_dict.get((from_layer, from_port), "?")

        edge_elements.append({
            'data': {
                'source': from_layer,
                'target': to_layer,
                'display_label': edge_shape_str
            }
        })

    return node_elements + edge_elements

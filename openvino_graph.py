import xml.etree.ElementTree as ET
from colors import BorderColorType

def parse_openvino_ir(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    layers_elem = root.find('layers')
    edges_elem = root.find('edges')

    if layers_elem is None or edges_elem is None:
        raise ValueError("Invalid IR structure: missing <layers> or <edges>")

    # Build shape dictionary from each layer's output ports.
    shape_dict = {}
    for layer in layers_elem.findall('layer'):
        layer_id = layer.get('id')
        output_elem = layer.find('output')
        if output_elem is not None:
            for port in output_elem.findall('port'):
                port_id = port.get('id')
                dims = [dim.text for dim in port.findall('dim')]
                shape_str = "x".join(dims) if dims else "?"
                shape_dict[(layer_id, port_id)] = shape_str

    # Build a mapping for each layer: layer_id -> {name, type}
    layer_info = {}
    for layer in layers_elem.findall('layer'):
        layer_id = layer.get('id')
        layer_info[layer_id] = {
            'name': layer.get('name'),
            'type': layer.get('type')
        }

    # Identify Const layers.
    const_ids = {lid for lid, info in layer_info.items() if info['type'] == 'Const'}

    # Identify Convert layers that have a Const parent.
    convert_ignore_ids = set()
    for edge in edges_elem.findall('edge'):
        from_layer = edge.get('from-layer')
        to_layer = edge.get('to-layer')
        if from_layer in const_ids:
            # If the target layer is a Convert layer, mark it to be ignored.
            if layer_info.get(to_layer, {}).get('type') == 'Convert':
                convert_ignore_ids.add(to_layer)

    # The complete set of layer IDs to ignore.
    ignore_node_ids = const_ids.union(convert_ignore_ids)

    # Build node elements, ignoring the nodes in ignore_node_ids.
    node_elements = []
    for layer in layers_elem.findall('layer'):
        layer_id = layer.get('id')
        if layer_id in ignore_node_ids:
            continue
        layer_type = layer.get('type')
        layer_name = layer.get('name')
        node_elements.append({
            'data': {
                'id': layer_id,
                'type': layer_type,
                'display_label': layer_type,
                'layer_name': layer_name,
                'border_color': BorderColorType.DEFAULT.value
            }
        })

    # Build edge elements; skip any edge connected to an ignored node.
    edge_elements = []
    for edge in edges_elem.findall('edge'):
        from_layer = edge.get('from-layer')
        from_port = edge.get('from-port')
        to_layer = edge.get('to-layer')
        if from_layer in ignore_node_ids or to_layer in ignore_node_ids:
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

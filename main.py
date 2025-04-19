import socket
from app import create_app

def get_local_ip():
    """ Helper to get local IP if you want to open from Windows side. """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def run_app():
    # ir_path = "/home/last1k/code/models/age-gender-recognition-retail-0013/age-gender-recognition-retail-0013.xml"
    ir_path = "/home/last1k/code/models/yolo_v8n/onnx/onnx/FP16/INT8/1/ov/optimized/yolo_v8n.xml"
    openvino_path = "/home/last1k/code/openvino/bin/intel64/Release"
    inputs_path = ["/home/last1k/images/Cat03.jpg"]
    # inputs_path = []

    app = create_app(openvino_path=openvino_path, ir_xml_path=ir_path, inputs_path=inputs_path)

    port = 8050
    local_ip = get_local_ip()
    url = f"http://{local_ip}:{port}"

    print(f"Starting Dash server. Visit {url} in your browser (or http://localhost:{port}).")
    app.run(debug=True, host='0.0.0.0', port=port)


if __name__ == '__main__':
    run_app()

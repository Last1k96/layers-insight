import socket
import argparse
import logging
import re
from app import create_app
from logger import log

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


def parse_arguments():
    parser = argparse.ArgumentParser(description='Layer Insight - OpenVINO Model Visualization Tool')
    parser.add_argument('--openvino_bin', required=True, help='Path to OpenVINO bin directory')
    parser.add_argument('--model', required=True, help='Path to model XML file')
    parser.add_argument('--inputs', required=True, help='Comma-separated list of input file paths')
    parser.add_argument('--port', type=int, default=8050, help='Port to start the server on (default: 8050)')
    parser.add_argument('--debug', action='store_true', help='Run the app in debug mode')

    return parser.parse_args()


def run_app(openvino_path, ir_path, inputs_path, port=8050, debug=False):
    # Convert comma-separated inputs to list if it's a string
    if isinstance(inputs_path, str):
        inputs_path = [path.strip() for path in inputs_path.split(',')]

    app = create_app(openvino_path=openvino_path, ir_xml_path=ir_path, inputs_path=inputs_path)

    # Disable annoying log messages when in a debug=False mode
    if not debug:
        logging.getLogger('werkzeug').disabled = True

    local_ip = get_local_ip()
    url = f"http://{local_ip}:{port}"

    print(f"Starting Dash server. Visit {url} in your browser (or http://localhost:{port}).")
    app.run(debug=debug, host='0.0.0.0', port=port)


if __name__ == '__main__':
    args = parse_arguments()

    # Log the parsed arguments
    log.info("Provided arguments:")
    log.info(f"  OpenVINO bin path: {args.openvino_bin}")
    log.info(f"  Model path: {args.model}")
    log.info(f"  Inputs path: {args.inputs}")
    log.info(f"  Port: {args.port}")

    run_app(
        openvino_path=args.openvino_bin,
        ir_path=args.model,
        inputs_path=args.inputs,
        port=args.port,
        debug=args.debug
    )

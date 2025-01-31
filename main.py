# run.py

import socket
import sys
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
    # Optional: read IR path from command line or hardcode
    # e.g. python run.py path/to/age_gender.xml
    ir_path = None
    if len(sys.argv) > 1:
        ir_path = sys.argv[1]

    app = create_app(ir_xml_path=ir_path)

    port = 8050
    local_ip = get_local_ip()
    url = f"http://{local_ip}:{port}"

    print(f"Starting Dash server. Visit {url} in your browser (or http://localhost:{port}).")
    app.run_server(debug=True, host='0.0.0.0', port=port)


if __name__ == '__main__':
    run_app()

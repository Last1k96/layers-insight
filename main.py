# run.py

import socket
from app import create_app


def get_local_ip():
    """
    Helper to get local IP if you want to open from Windows side.
    Usually 'localhost' or '127.0.0.1' also works for WSL usage.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def main():
    # Create the Dash app (optionally pass IR file path)
    app = create_app()

    port = 8050
    host_ip = get_local_ip()
    url = f"http://{host_ip}:{port}"

    print(f"Starting Dash server. Visit {url} in your browser, or use http://localhost:{port}.")

    # Run on 0.0.0.0 to allow external access (e.g. from Windows)
    app.run_server(debug=True, host='0.0.0.0', port=port)


if __name__ == '__main__':
    main()

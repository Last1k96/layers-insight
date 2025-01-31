from flask import Flask, render_template
import webbrowser
import socket

app = Flask(__name__)


@app.route("/")
def index():
    """
    Render our main page that shows the sample graph.
    """
    return render_template("index.html")


def get_local_ip():
    """
    Simple helper to get the local network IP address if you want to
    open the page from the Windows side. For WSL, you can also use 'localhost'.
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


if __name__ == "__main__":
    # Choose a port to run on
    port = 5000
    url = f"http://{get_local_ip()}:{port}"

    print(f"Server starting. Go to {url} in your browser.")

    # Optionally, open automatically in browser (sometimes fails in WSL)
    # webbrowser.open(url)

    app.run(host="0.0.0.0", port=port, debug=True)

from setuptools import setup, find_packages

setup(
    name="LayerInsight",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "dash",
        "dash-cytoscape",
        "dash-bootstrap-components",
        "dash-extensions",
        "dash-split-pane",
        "numpy",
        "opencv-python-headless",
        "scikit-image",
        "plotly",
        "matplotlib",
        "scipy",
        "umap-learn",
        "colorlog"
    ],
)

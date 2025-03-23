from setuptools import setup, find_packages

setup(
    name="your_project_name",  # Replace with your project's name
    version="0.1.0",           # Replace with your project's version
    packages=find_packages(),
    install_requires=[
        "dash==3.0.0",
        "dash-cytoscape",
        "dash-bootstrap-components",
        "dash-extensions",
        "dash-split-pane",
        "numpy",
        "opencv-python-headless",
        "scikit-image~=0.25.1",
        "plotly~=6.0.0",
        "matplotlib~=3.10.0",
        "scipy~=1.15.1",
        "umap-learn"
    ],
)
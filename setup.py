from setuptools import find_packages, setup

setup(
    name="agentic-integrator",
    version="0.1.0",
    description="Agent-S3 + World Model: Human-Like Autonomous Screen Navigation",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="jeevesh415",
    url="https://github.com/jeevesh415/agentic-integrator",
    packages=find_packages(),
    install_requires=[
        "gui-agents>=0.3.0",
        "openai>=1.0.0",
        "anthropic>=0.20.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "dataclasses-json>=0.6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "black", "mypy"],
        "diffusion": ["torch>=2.0.0", "diffusers>=0.25.0"],
    },
    entry_points={
        "console_scripts": [
            "agentic-integrator=agentic_integrator.cli:main",
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

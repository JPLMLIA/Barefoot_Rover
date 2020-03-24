from setuptools import setup, find_packages

long_description = ""
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="barefoot",
    version="0.0.1",
    # author="Example Author",
    # author_email="author@example.com",
    # description="Barefoot",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github-fn.jpl.nasa.gov/BarefootRover/Barefoot_Rover",
    entry_points={
        "console_scripts": ["togawrapper = toga_wrapper:main"]

    },
    packages=find_packages(),
    tests_require=['pytest'],
    python_requires='>=3.7',
)
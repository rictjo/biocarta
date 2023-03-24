import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name         = "biocarta",
    version      = "0.2.26",
    author       = "Richard Tjörnhammar",
    author_email = "richard.tjornhammar@gmail.com",
    description  = "",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/rictjo/biocarta",
    packages = setuptools.find_packages('src'),
    package_dir = {'biocarta':'src/biocarta','quantification':'src/quantification','special':'src/special','enrichment':'src/enrichment'},
    classifiers = [
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)

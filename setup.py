import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name         = "biocartograph",
    version      = "0.9.0",
    author       = "Richard Tjörnhammar",
    author_email = "richard.tjornhammar@gmail.com",
    description  = "Package was renamed from Biocarta v0.2.27 to Biocartograph because of an unintentional name clash",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/rictjo/biocarta",
    packages = setuptools.find_packages('src'),
    package_dir = {	'biocartograph':'src/biocartograph' , 'quantification':'src/quantification' , 'models':'src/models',
			'composition':'src/composition' , 'special':'src/special' , 'enrichment':'src/enrichment' },
    classifiers = [
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)

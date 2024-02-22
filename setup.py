import setuptools

with open("README.md" , "r", encoding = "utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "Medicinal-Leaf-Classification"
AUTHOR_USERNAME = "chaitanya-kolliboyina"
SRC_REPO = "MedicineLeaf-Classifier"
AUTHOR_EMAIL = "kvschaitanya17@gmail.com"

setuptools.setup(
    name = SRC_REPO,
    version = __version__,
    author = AUTHOR_USERNAME,
    author_email = AUTHOR_EMAIL,
    description = "python package for DL app to classify medicinal leaf",
    long_description=long_description,
    long_description_content = "text/markdown",
    url = f"https://github.com/{AUTHOR_USERNAME}/{REPO_NAME}",
    project_urls = {
        "Bug_tracker":f"https://github.com/{AUTHOR_USERNAME}/{REPO_NAME}/issues",
    },
    package_dir = {"":"src"},
    packages = setuptools.find_packages(where = "src"),
 )
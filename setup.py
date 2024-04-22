from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name= "PipeLineProject",
    version= "0.1.0",
    author= "Somdeep Sardar",
    author_email= "sardarsomdeep123@gmail.com",
    description= "An End-to-End Machine Learning Project using Python and Pipeline.",
    install_requires = get_requirements("requirements.txt"),
    packages= find_packages()
)
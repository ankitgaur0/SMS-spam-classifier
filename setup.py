from setuptools import setup,find_packages
from typing import List
from pathlib import Path

#This variable define in the Requirements.txt file.
HYPEN_e_DOT="-e ."


def get_requirements(file_path :str) ->List[str]:
    # make a list of packages names
    packages=[]
    file_path=Path(file_path)
    with open(file_path,"r") as file:
        packages=file.readlines()


    # packages store the escape sequence character like \n .remove it.
    packages=[a.replace("\n","") for a in packages]


    # -e . remove in the list
    if HYPEN_e_DOT in packages:
        packages=packages.remove(HYPEN_e_DOT)

    return packages

setup(
    name="SMS_spam_classification",
    version='0.0.1',
    description="This modle will determine a sms that it is spam or ham",
    author="ANKIT GAUR",
    author_email="ankitparashar000@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("Requirements.txt")
)
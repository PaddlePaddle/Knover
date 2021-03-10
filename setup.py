#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Setup Knover."""

import setuptools


with open("README.md", "r") as f:
    readme = f.read()


if __name__ == "__main__":
    print(setuptools.find_packages())
    setuptools.setup(
        name="knover-dygraph",
        version="0.1.0",
        description="Large-scale open domain KNOwledge grounded conVERsation system based on PaddlePaddle",
        long_description=readme,
        long_description_content_type="text/markdown",
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3.7",
            "License :: OSI Approved :: Apache Software License",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        python_requires=">=3.7",
        install_requires=[
            "paddlepaddle-gpu>=2.0.1",
            "numpy",
            "sentencepiece",
            "termcolor",
            "tqdm"
        ]
    )

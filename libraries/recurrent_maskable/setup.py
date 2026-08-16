from setuptools import setup, find_packages

if __name__ == '__main__':
    setup(
        name="recurrent_maskable",
        version="0.1.0",
        packages=find_packages(),
        py_modules=["policies", "ppo_mask_recurrent"],
    )


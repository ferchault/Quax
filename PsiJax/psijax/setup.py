import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name='psijax',
        version="0.1.1",
        description='Arbitrary order derivatives of electronic structure computations.',
        author='Adam Abbott',
        author_email='adabbott@uga.edu',
        url="none",
        license='BSD-3C',
        packages=setuptools.find_packages(),
        install_requires=[
            'numpy>=1.7','jax>=0.1.70','jaxlib>=0.1.50'
        ],
        extras_require={
            'tests': [
                'pytest-cov',
            ],
        },

        tests_require=[
            'pytest-cov',
        ],

        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
        ],
        zip_safe=True,
    )

from setuptools import setup, find_packages

scripts = [
    'bin/run-metadetect-on-slices',
]

setup(
    name='pizza_cutter_metadetect',
    description="run metadetect on yummy survey slices",
    author="MRB and ESS",
    packages=find_packages(),
    include_package_data=True,
    scripts=scripts,
    use_scm_version=True,
    setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
)

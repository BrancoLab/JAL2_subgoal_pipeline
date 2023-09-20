"""

Setuptools is a collection of enhancements to the Python distutils that allow developers 
to more easily build and distribute Python packages, especially ones that have dependencies on other packages.
Packages built and distributed using setuptools look to the user like ordinary Python packages based on the distutils

"""

#OS Libaries
import setuptools

setuptools.setup(
    name='behave-analysis',
    version='1.3',
    author='Philip Shamash',
    license='GNU General Public License',
    packages = ['behave_analysis'],
    package_dir={'behave_analysis': 'behave_analysis'},
    entry_points={
        "console_scripts": [
            "process = behave_analysis.run:process",
            "analyze = behave_analysis.run:analyze",
            "track = behave_analysis.run:track",
            "visualize = behave_analysis.run:visualize",    
            "homings = behave_analysis.run:homings"       
        ]
    }
)
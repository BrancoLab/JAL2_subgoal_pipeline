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
            "process = behave_analysis.run.run_process:process",
            "postprocess = behave_analysis.run.run_postprocess:postprocess",
            "analyze_efizz = behave_analysis.run.run_analyze_efizz:analyze_efizz",
            "analyze_behave = behave_analysis.run.run_analyze_behave:analyze_behave",
            "track = behave_analysis.run.run_track:track",
            "visualize = behave_analysis.run.run_visualize:visualize",
        ]
    }
)
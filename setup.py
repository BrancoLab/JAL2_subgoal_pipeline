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
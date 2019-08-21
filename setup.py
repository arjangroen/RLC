from setuptools import setup

setup(
    name='RLC',
    version='0.3',
    packages=['RLC', 'RLC.move_chess', 'RLC.capture_chess', 'RLC.real_chess'],
    url='https://github.com/arjangroen/RLC',
    license='MIT',
    author='a.groen',
    author_email='arjanmartengroen@gmail.com',
    description='Collection of reinforcement learning algorithms, all applied to chess or chess related problems'
)

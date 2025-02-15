from setuptools import setup, find_packages

setup(
    name="drone_rl",
    version="0.1.0",  # 版本号与 package.xml 保持一致
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=["rospy", "gym", "stable-baselines3"]
)

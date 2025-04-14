cpp:
	cd trade && python setup.py build_ext && cd ..  && find trade/build/ -name "*.so" -exec cp -f {} trade/ \;

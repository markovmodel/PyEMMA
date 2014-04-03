To run emma2 you need the following dependencies being installed.
You get binary distributions of these in dmg packages, which can be easily 
installed. Note that you may need admin privilegdes to do so.

0. install an Java JDK (>= 1.6) from http://oracle.com
1. install python-2.7 from python.org; run "Update shell Profile.command" to 
update your PATH to point to the installation of new python
1a. If you set a JAVA_HOME environment variable to point to your JDK and use
 sudo to install jpype (inclusive in step 5), make sure to use sudo -E to
 preserve the JAVA_HOME variable!
2. install numpy (latest) from http://numpy.org
3. install scipy (latest) from http://scipy.org 

4. If you got that, you need to make sure, you own a C/C++ compiler.
There is one bundled with Xcode, but you need an Apple Developer ID for that.
Try: https://github.com/kennethreitz/osx-gcc-installer/ if you do not want such
an ID nor have one. 

5. install emma2 via
	python setup.py install [--user]

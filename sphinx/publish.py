#!/usr/bin/env python

##########################################################################################
#
#	publish.py
#
#	You don't need to use this script. This is not the script you are looking for.
#
#                              Chris Wehmeyer <christoph.wehmeyer@fu-berlin.de>
#
##########################################################################################

import sys
import argparse
from subprocess import call

htmlbuild = "_build/html"
userpage = "/web/page.mi.fu-berlin.de/web-home/schiffm/public_html"
emmadoc = "emma-doc"

target = userpage+"/"+emmadoc

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument( "--userpage", help="publish the html build on my userpage", action="store_true" )
	args = parser.parse_args()

	if args.userpage:
		try:
			call( [ "rm", "-rf", target+'/*' ] )
		except:
			print "Cannot delete '"+target+"/*'!"
			sys.exit( 1 )
		try:
			call( [ "cp", "-r", htmlbuild+'/*', target+"/" ] )
		except:
			print "Cannot copy files to '"+target+"/*'!"
			sys.exit( 1 )
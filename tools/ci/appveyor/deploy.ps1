function deploy() {
    if (-not ($env:APPVEYOR_REPO_TAG)) { # no new tag
        #return
    }
    # install tools
    pip install wheel twine
    
    # create wheel and win installer
    python setup.py bdist_wheel bdist_wininst
    
    # upload to pypi with twine
    twine upload -i $env:myuser -p $env:mypass dist/*
}
if [ "$TRAVIS_PULL_REQUEST" = true ]; then
    echo "This is a pull request. No deployment will be done."; exit 0
fi


if [ "$TRAVIS_BRANCH" != "devel" ]; then
    echo "No deployment on BRANCH='$TRAVIS_BRANCH'"; exit 0
fi


# Deploy to binstar
conda install --yes anaconda-client 
anaconda -t $BINSTAR_TOKEN upload --force -u omnia -p ${PACKAGENAME}-dev $HOME/miniconda/conda-bld/*/${PACKAGENAME}-dev-*.tar.bz2

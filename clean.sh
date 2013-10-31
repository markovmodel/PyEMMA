# deletes build artifacts
rm build -rf
rm dist -rf
rm *egg-info -rf
rm temp -r
# delete python byte codes
find -name *.pyc -exec rm {} +

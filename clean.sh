# deletes build artifacts
rm build -rf
rm dist -rf
rm *egg-info -rf
rm temp -rf
# delete python byte codes
find -name *.pyc -exec rm {} +
find -name *.log -exec rm {} +

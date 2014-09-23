# deletes build artifacts
rm target -rf
rm build -rf
rm dist -rf
rm *egg-info -rf
rm *.egg -rf
rm temp -rf
# delete python byte codes
find -name *.pyc -exec rm {} +
find -name *.log -exec rm {} +
# delete shared objects (linux)
find -name *.so -exec rm {} +

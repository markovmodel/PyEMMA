Developer Notes
===============


Publish a new release
---------------------
1. merge current devel branch into master
   git checkout master; git merge devel
2. make a new tag 'vmajor.minor.patch' where major is major release and so on
   git tag -m "release description" v1.1
3. IMPORTANT: first push, then push --tags
   git push; git push --tags

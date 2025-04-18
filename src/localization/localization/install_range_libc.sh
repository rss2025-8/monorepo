#!/bin/bash
git clone https://github.com/Giantpizzahead/range_libc ~/range_libc
cd ~/range_libc/pywrapper
echo "Compiling range_libc (may ask for sudo password...)"
./compile.sh
rm -rf ~/range_libc
echo "Done!"

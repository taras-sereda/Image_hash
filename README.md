Image_hash
==========
#image_hash
Creates 64 bit Image hashs, could be used for finding duplicates, comparing images using Hamming distance.

### Dependencies
cmake 2.8 
OpenCV 2.4.9

## Building
unzip image_hash.zip
cd image_hash
mkdir build
cd build
cmake ..
make

## Running
./image_hash Images_dir

mkdir -p data
cd data

## Download real testing dataset
url="https://www.dropbox.com/s/hdnbh526tyvv68i/DiLiGenT.zip?dl=0"
name="DiLiGenT"

#wget $url -O ${name}.zip
#unzip ${name}.zip
#rm ${name}.zip

cd ${name}/pmsData/
cp ballPNG/filenames.txt names.txt

# Back to root directory
cd ../../../
cp scripts/DiLiGenT_objects.txt data/${name}/pmsData/objects.txt
python3 scripts/cropDiLiGenTDataHDR.py

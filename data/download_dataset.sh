 
#!/bin/bash
CURDIR=$(pwd)
if [ -d $CURDIR/RAMS_1.0 ]
then 
rm -Rf $CURDIR/RAMS_1.0
fi

if [ -d $CURDIR/WikiEvent ]  
then    
	rm -Rf $CURDIR/WikiEvent
fi


if [[ "$CURDIR" =~ "PAIE/data" ]]
then
	echo "please run this script under the root dir of the project, eg directory PAIE"
	echo "please input the command ' cd .. ' then press return  "
	exit -1
else

	echo "downloading data from a server ... "

fi
# Download RAMS
wget -c https://nlp.jhu.edu/rams/RAMS_1.0b.tar.gz
tar -zxvf ./RAMS_1.0b.tar.gz
rm -rf ./RAMS_1.0b.tar.gz
mv ./RAMS_1.0 ./data/

# Download WIKIEVENTS
WIKIDIR=./data/WikiEvent/data/
mkdir -p $WIKIDIR
wget -c -P $WIKIDIR  https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data/train.jsonl
wget -c -P $WIKIDIR  https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data/dev.jsonl
wget -c -P $WIKIDIR  https://gen-arg-data.s3.us-east-2.amazonaws.com/wikievents/data/test.jsonl
tree data

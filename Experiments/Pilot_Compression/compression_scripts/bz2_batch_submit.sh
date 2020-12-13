#$ -S /bin/sh
#$ -cwd
#$ -j y

export PATH=/colossus/home/arnon/1000Genomes:$PATH
source /share/apps/setpath.sh

for FILE in Data/Quality_Small_Batch/*.quality
do
	echo $FILE

	for i in {1..10}
	do
		FILE_NAME="${FILE##*/}"

   	    	 /usr/bin/time -f "%e,%S,%U,%P,%K,%M" bzip2 -c $FILE > Data/Temp_Bz2/$FILE_NAME.bz2
        	stat -c %s Data/Temp_Bz2/$FILE_NAME.bz2
        	/usr/bin/time -f "%e,%S,%U,%P,%K,%M" bzip2 -cd Data/Temp_Bz2/$FILE_NAME.bz2 > Data/Temp_Bz2/$FILE_NAME.quality
		
		rm Data/Temp_Bz2/$FILE_NAME*
	done
done


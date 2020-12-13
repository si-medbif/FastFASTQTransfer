#$ -S /bin/sh
#$ -cwd
#$ -j y

export PATH=/colossus/home/arnon/1000Genomes:$PATH

for FILE in Data/Quality_Small_Batch/*.quality
do
        echo $FILE

        for i in {1..10}
        do
		FILE_NAME="${FILE##*/}"

		/usr/bin/time -f "%e,%S,%U,%P,%K,%M" gzip -c $FILE > Data/Temp_Gzip/$FILE_NAME.gz
		stat -c %s Data/Temp_Gzip/$FILE_NAME.gz
		/usr/bin/time -f "%e,%S,%U,%P,%K,%M" gzip -cd Data/Temp_Gzip/$FILE_NAME.gz > Data/Temp_Gzip/$FILE_NAME.quality
				
        done
done

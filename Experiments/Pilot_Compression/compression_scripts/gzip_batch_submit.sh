for FILE in Data/Quality/*.quality
do
	echo $FILE
	for i in {1..10} ;
	do
		FILE_NAME="${FILE##*/}"
		
		touch Results/Pilot_Compression/gzip_compression_stat_$FILE_NAME.result
 
		/usr/bin/time -f "%E,%U,%K,%M" -a -o Results/Pilot_Compression/gzip_compression_stat_$FILE_NAME.result gzip -kc $FILE > Data/Temp/$FILE_NAME.gz

		stat -c %s Data/Temp/$FILE_NAME.gz >> Results/Pilot_Compression/gzip_compression_stat_size_$FILE_NAME.result

		/usr/bin/time -f "%E,%U,%K,%M" -a -o Results/Pilot_Compression/gzip_decompression_stat_$FILE_NAME.result gzip -fd Data/Temp/$FILE_NAME.gz

		rm Data/Temp/$FILE_NAME*
	done
done

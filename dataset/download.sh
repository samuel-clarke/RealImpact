while read -r line; do
    echo $line
    wget http://downloads.cs.stanford.edu/viscam/RealImpact/$line.zip
    unzip $line.zip
    rm $line.zip
done < object_names.txt
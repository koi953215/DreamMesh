# Iterate over all glb files in the out directory
for file in out/*.glb; do

    filename=$(basename -- "$file")
    name="${filename%.*}"
    echo "Name: $name"

    ./launch/inference/generate_skeleton.sh --input out/${filename}  --output results/${name}.fbx
    ./launch/inference/generate_skin.sh --input results/${name}.fbx --output out/${name}_skin.fbx
done


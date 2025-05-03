cd src/main/resources/cu/
for file in *.cu; do
    nvcc -m64 -ptx "$file" -o "${file%.cu}.ptx"
done
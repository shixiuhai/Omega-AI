# 进入目标目录
cd src/main/resources/cu/

# 获取所有 .cu 文件并逐个处理
Get-ChildItem *.cu | ForEach-Object {
    $file = $_.FullName
    $outputFile = [System.IO.Path]::ChangeExtension($file, ".ptx")
    nvcc -m64 -ptx $file -o $outputFile
}
# 进入目标目录
cd src/main/resources/cu/

# 遍历所有 .cu 文件
for file in *.cu; do
    # 获取文件完整路径
    full_path=$(pwd)/$file
    # 构造输出文件名
    output_file="${full_path%.cu}.ptx"
    # 执行编译命令
    nvcc -m64 -ptx $full_path -o $output_file
done

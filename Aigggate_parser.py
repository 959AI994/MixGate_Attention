import os
import re
from glob import glob
from collections import defaultdict


def append_to_bench_file(directory, filename, content):
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"{filename}.bench")
    if os.path.exists(file_path):
        with open(file_path, 'a') as file:
            file.write(content)
    else:
        with open(file_path, 'w') as file:
            file.write(content)


def replace_nodes(file_content, dic_nodes):
    pattern = r'\bn(\d+)|\bpo(\d+)|\ba(\d+)'
    for i, line in enumerate(file_content):
        if isinstance(line, str):
            def replace(match):
                full_key = match.group(0)
                return str(dic_nodes.get(full_key, full_key))
            file_content[i] = re.sub(pattern, replace, line)


def extract_lines_from_files(directory, aimed_directory):
    """
    使用 glob 递归提取目录中的文件内容，并保留子目录结构
    """
    dic = {"0xe8": [0, 0, 0], "0x8e": [1, 0, 0], "0xb2": [0, 1, 0], "0xd4": [0, 0, 1]}

    # 使用 glob 遍历所有子目录和文件
    file_paths = glob(f"{directory}/**/*", recursive=True)

    for file_path in file_paths:
        if os.path.isfile(file_path):  # 确保是文件
            # 获取相对路径和目标文件路径
            relative_path = os.path.relpath(file_path, directory)
            aimed_file_path = os.path.join(aimed_directory, relative_path)
            aimed_subdir = os.path.dirname(aimed_file_path)

            # 创建目标子目录
            if not os.path.exists(aimed_subdir):
                os.makedirs(aimed_subdir)

            # 文件处理逻辑
            file_content = []
            added_node_dic = []
            added_nodes = 0
            count_input = 1
            no_nodes = 1
            dic_nodes = {}

            with open(file_path, 'r') as file:
                dic_nodes["n0"] = 0
                lines = file.readlines()
                file_content.append("INPUT(n0)\n")
                for line in lines:
                    if "LUT" not in line:
                        file_content.append(line)
                        if "INPUT" in line:
                            input_node = line.split("(")[1].split(")")[0]
                            dic_nodes[input_node] = no_nodes
                            count_input += 1
                            no_nodes += 1
                        elif "OUTPUT" in line:
                            output_node = line.split("(")[1].split(")")[0]
                            dic_nodes[output_node] = no_nodes
                            no_nodes += 1
                    else:
                        if "po" not in line:
                            inner_node = line.split(" ")[0]
                            dic_nodes[inner_node] = no_nodes
                            no_nodes += 1
                        if "0xe8" in line:
                            content = line.split(" ")
                            file_content.append(content[0] + " = " + "MAJ" + content[4] + " " + content[5] + " " + content[6])
                        elif "0x8e" in line:
                            content = line.split(" ")
                            added_node_dic.append(f"a{added_nodes}")
                            file_content.append(content[0] + " = " + "MAJ" + "(" + added_node_dic[added_nodes] + "," + " " + content[5] + " " + content[6])
                            node = line.split("(")[1].split(",")[0]
                            file_content.append(added_node_dic[added_nodes] + " = " + "NOT" + "(" + node + ")" + '\n')
                            added_nodes += 1
                        elif "0xb2" in line:
                            content = line.split(" ")
                            added_node_dic.append(f"a{added_nodes}")
                            file_content.append(content[0] + " = " + "MAJ" + content[4] + " " + added_node_dic[added_nodes] + "," + " " + content[6])
                            node = line.split("(")[1].split(",")[1].split(" ")[1]
                            file_content.append(added_node_dic[added_nodes] + " = " + "NOT" + "(" + node + ")" + '\n')
                            added_nodes += 1
                        elif "0xd4" in line:
                            content = line.split(" ")
                            added_node_dic.append(f"a{added_nodes}")
                            file_content.append(content[0] + " = " + "MAJ" + content[4] + " " + content[5] + " " + added_node_dic[added_nodes] + ")" + '\n')
                            node = line.split("(")[1].split(" ")[2].split(")")[0]
                            file_content.append(added_node_dic[added_nodes] + " = " + "NOT" + "(" + node + ")" + '\n')
                            added_nodes += 1
                        elif "0x96" in line:
                            content = line.split(" ")
                            file_content.append(content[0] + " = " + "XOR" + content[4] + " " + content[5] + " " + content[6])
                        elif "0x8" in line:
                            content = line.split(" ")
                            file_content.append(content[0] + " = " + "AND" + content[4] + " " + content[5])
                        elif "0x6" in line:
                            content = line.split(" ")
                            file_content.append(content[0] + " = " + "XOR" + content[4] + " " + content[5])
                        elif "0x4" in line:
                            content = line.split(" ")
                            added_node_dic.append(f"a{added_nodes}")
                            file_content.append(content[0] + " = " + "AND" + content[4] + " " + added_node_dic[added_nodes] + ")" + '\n')
                            node = line.split("(")[1].split(" ")[1].split(")")[0]
                            file_content.append(added_node_dic[added_nodes] + " = " + "NOT" + "(" + node + ")" + '\n')
                            added_nodes += 1
                        elif "0x2" in line and len(line.split(" ")) >= 6:
                            content = line.split(" ")
                            added_node_dic.append(f"a{added_nodes}")
                            file_content.append(content[0] + " = " + "AND" + "(" + added_node_dic[added_nodes] + "," + " " + content[5])
                            node = line.split("(")[1].split(" ")[0].split(",")[0]
                            file_content.append(added_node_dic[added_nodes] + " = " + "NOT" + "(" + node + ")" + '\n')
                            added_nodes += 1
                        elif "0x1" in line and len(line.split(" ")) <= 5:
                            content = line.split(" ")
                            added_node_dic.append(f"a{added_nodes}")
                            file_content.append(content[0] + " = " + "NOT" + content[4])
                            added_nodes += 1
                        elif "0x1" in line and len(line.split(" ")) >= 6:
                            content = line.split(" ")
                            added_node_dic.append(f"a{added_nodes}")
                            added_nodes += 1
                            added_node_dic.append(f"a{added_nodes}")
                            file_content.append(content[0] + " = " + "AND" + "(" + added_node_dic[added_nodes - 1] + ", " + added_node_dic[added_nodes] + ")" + '\n')
                            node_0 = line.split(" ")[4].split("(")[1].split(",")[0]
                            node_1 = line.split(" ")[5].split(")")[0]
                            file_content.append(added_node_dic[added_nodes - 1] + " = " + "NOT" + "(" + node_0 + ")" + '\n')
                            file_content.append(added_node_dic[added_nodes] + " = " + "NOT" + "(" + node_1 + ")" + '\n')
                            added_nodes += 1

                for i in range(added_nodes):
                    dic_nodes[f"a{i}"] = no_nodes
                    no_nodes += 1

                replace_nodes(file_content, dic_nodes)

                if len(dic_nodes) > 1000:
                    continue

                # 写入目标文件
                with open(aimed_file_path, 'w') as aimed_file:
                    for item in file_content:
                        if item != '0 = gnd\n':
                            aimed_file.write(item)


if __name__ == "__main__":
    
    directory = '/home/wjx/npz/raw_aig/sub_bench'
    aimed_directory = '/home/wjx/npz/aimed_aig'

    if not os.path.exists(aimed_directory):
        os.makedirs(aimed_directory)

    extract_lines_from_files(directory, aimed_directory)
